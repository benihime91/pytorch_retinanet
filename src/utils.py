import math
from typing import *
import torch
import torch.nn.functional as F
from torch.functional import Tensor
from torchvision.ops.boxes import box_iou
from .config import *
from .utilities import ifnone
from .losses import focal_loss, smooth_l1_loss


def bbox_2_activ(ground_truth_boxes: Tensor, anchors: Tensor) -> Tensor:
    """
    Convert `ground_truths` to match the model `activations` to calculate `loss`.
    """
    if anchors.device != ground_truth_boxes.device:
        anchors.to(ground_truth_boxes.device)
    
    # Unpack Elements
    anchors_x1 = anchors[:, 0].unsqueeze(1)
    anchors_y1 = anchors[:, 1].unsqueeze(1)
    anchors_x2 = anchors[:, 2].unsqueeze(1)
    anchors_y2 = anchors[:, 3].unsqueeze(1)

    ground_truths_x1 = ground_truth_boxes[:, 0].unsqueeze(1)
    ground_truths_y1 = ground_truth_boxes[:, 1].unsqueeze(1)
    ground_truths_x2 = ground_truth_boxes[:, 2].unsqueeze(1)
    ground_truths_y2 = ground_truth_boxes[:, 3].unsqueeze(1)

    # Calculate width, height, center_x, center_y
    w = anchors_x2 - anchors_x1
    h = anchors_y2 - anchors_y1
    x = anchors_x1 + 0.5 * w
    y = anchors_y1 + 0.5 * h

    gt_w = ground_truths_x2 - ground_truths_x1
    gt_h = ground_truths_y2 - ground_truths_y1
    gt_x = ground_truths_x1 + 0.5 * gt_w
    gt_y = ground_truths_y1 + 0.5 * gt_h

    # Calculate Offsets
    dx = BBOX_REG_WEIGHTS[0] * (gt_x - x) / w
    dy = BBOX_REG_WEIGHTS[1] * (gt_y - y) / h
    dw = BBOX_REG_WEIGHTS[2] * torch.log(gt_w / w)
    dh = BBOX_REG_WEIGHTS[3] * torch.log(gt_h / h)

    targets = torch.cat((dx, dy, dw, dh), dim=1)
    return targets


def activ_2_bbox(activations: Tensor, anchors: Tensor, clip_activ: float = math.log(1000.0 / 16)) -> Tensor:
    "Converts the `activations` of the `model` to bounding boxes."

    # Gather in the same device
    if anchors.device != activations.device:
        anchors = anchors.to(activations.device)

    # Convert anchor format from XYXY to XYWH
    w = anchors[:, 2] - anchors[:, 0]
    h = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * w
    ctr_y = anchors[:, 1] + 0.5 * h

    # Calculate Offsets
    dx = activations[:, 0::4] / BBOX_REG_WEIGHTS[0]
    dy = activations[:, 1::4] / BBOX_REG_WEIGHTS[1]
    dw = activations[:, 2::4] / BBOX_REG_WEIGHTS[2]
    dh = activations[:, 3::4] / BBOX_REG_WEIGHTS[3]

    # Clip offsets
    dw = torch.clamp(dw, max=clip_activ)
    dh = torch.clamp(dh, max=clip_activ)

    # Extrapolate bounding boxes on anchors from the model activations.
    pred_ctr_x = dx * w[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * h[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]

    # Convert bbox shape from xywh to x1y1x2y2
    pred_boxes1 = (pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w)
    pred_boxes2 = (pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h)
    pred_boxes3 = (pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w)
    pred_boxes4 = (pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h)

    # Stack the co-ordinates to the bbox co-ordinates.
    pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
    return pred_boxes


def matcher(
    targets: Tensor, anchors: Tensor, match_thr: float = None, back_thr: float = None,
) -> Tensor:
    """
    Match `anchors` to targets. -1 is match to background, -2 is ignore.
    """
    # From: https: // github.com/fastai/course-v3/blob/9c83dfbf9b9415456c9801d299d86e099b36c86d/nbs/dl2/pascal.ipynb
    # - for each anchor we take the maximum overlap possible with any of the targets.
    # - if that maximum overlap is less than 0.4, we match the anchor box to background,
    # the classifier's target will be that class.
    # - if the maximum overlap is greater than 0.5, we match the anchor box to that ground truth object.
    # The classifier's target will be the category of that target.
    # - if the maximum overlap is between 0.4 and 0.5, we ignore that anchor in our loss computation.

    match_thr = ifnone(match_thr, IOU_THRESHOLDS_FOREGROUND)
    back_thr = ifnone(back_thr, IOU_THRESHOLDS_BACKGROUND)

    assert (
        match_thr > back_thr
    ), "Threshold for `match` should be greater than `background`"
    # Calculate IOU between given targets & anchors
    iou_vals = box_iou(targets, anchors)

    # Grab the best ground_truth overlap
    vals, idxs = iou_vals.max(dim=0)

    # Assign candidate matches with low quality to negative (unassigned) values
    # Threshold less than `back_thr` gets assigned -1 : background
    idxs[vals < back_thr] = torch.tensor(BACKGROUND_IDX)

    # Threshold between `match_thr` & `back_thr` gets assigned -2: ignore
    idxs[(vals >= back_thr) & (vals < match_thr)] = torch.tensor(IGNORE_IDX)
    return idxs


def retinanet_loss(targets: List[Dict[str, Tensor]], outputs: Dict[str, Tensor], anchors: List[Tensor]) -> Dict[str, Tensor]:
    """
    Loss for the `classification subnet` & `regression subnet` of `RetinaNet`
    """
    # ---------------------------------------------------------------------
    # Calculate matched idxs & convert `targets` (bboxes) to activations.
    # ---------------------------------------------------------------------
    matched_idxs = []
    for ancs, targs in zip(anchors, targets):
        if targs['boxes'].numel() == 0:
            matched_idxs.append(torch.empty((0,), dtype=torch.int32))
            continue
        matched_idxs.append(bbox_2_activ(targs["boxes"], ancs))


    # ---------------------------------------------------------------------
    # Instantiate `vars`
    # ---------------------------------------------------------------------
    # matched_idxs will contain the values `cls_id`, -1 & -2 for
    # foreground_cls(valid_classes), background_cls & cls_to_be_ignored
    # Outputs is the output of the `RetinanetHead` which is a dictionary
    # Grab the cls_logits from the outputs and bbox_reressions from the Ouputs
    cls_logits = outputs["logits"]
    bbox_regression = outputs["bboxes"]

    # ---------------------------------------------------------------------
    # Calculate Classification Loss
    # ---------------------------------------------------------------------
    classification_loss = torch.tensor(0.0)
    for cls_tgt, cls_pred, matches in zip(targets, cls_logits, matched_idxs):
        # no matched_idxs means there were no annotations in this image
        if matches.numel() == 0:
            gt_class_target = torch.zeros_like(cls_pred)
            valid_idxs_per_image = torch.arange(cls_pred.shape[0])
            num_foregrounds = torch.tensor(0.0)

        else:
            # Extract the idxs of the foreground classes
            class_mask = matches >= 0
            num_foregrounds = class_mask.sum()
            # Create the classification targets
            # one_hot encode the classification targets
            gt_class_target = torch.zeros_like(cls_pred)
            gt_class_target = [class_mask,cls_tgt["labels"][matches[class_mask]],] = torch.tensor(1.0)
            # Find Indices where anchors should be ignored
            valid_idxs_per_image = matches != IGNORE_IDX

        valid_cls_logits = cls_pred[valid_idxs_per_image]
        valid_gt_class_target = gt_class_target[valid_idxs_per_image]

        # Compute Focal Loss
        classification_loss += focal_loss(valid_cls_logits, valid_gt_class_target, reduction="sum") / max(1, num_foregrounds)

    # ---------------------------------------------------------------------
    # Calculate Regression Loss
    # ---------------------------------------------------------------------
    bbox_regress_loss = torch.tensor(0.0)
    for bbox_tgt, bbox_pred, ancs, matches in zip(targets, bbox_regression, anchors, matched_idxs):
        # no matched_idxs means there were no annotations in this image
        if matches.numel() == 0:
            continue

        # get the targets corresponding GT for each proposal
        matched_gt_boxes_per_image = bbox_tgt["boxes"][matches.clamp(min=0)]
        # determine only the foreground indices, ignore the rest
        bbox_mask = matches >= 0
        num_foreground = bbox_mask.sum()
        # select only the foreground boxes
        matched_gt_boxes_per_image = matched_gt_boxes_per_image[bbox_mask, :]
        bbox_pred_per_image = bbox_pred[bbox_mask, :]
        ancs_per_image = ancs[bbox_mask, :]
        # Encode the `gt_bboxes` to `activations`
        target_regression = bbox_2_activ(matched_gt_boxes_per_image, ancs_per_image)
        # compute the loss
        bbox_regress_loss += smooth_l1_loss(bbox_pred_per_image, target_regression, reduction="sum") / max(1, num_foreground)

        loss_dict = {
        "classification_loss": classification_loss / max(1, len(targets)),
        "bbox_regression_loss": bbox_regress_loss / max(1, len(targets)),}

        return loss_dict
