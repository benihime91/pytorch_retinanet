import math
from typing import *
import torch
from torch.functional import Tensor
from torchvision.ops.boxes import box_iou
from .config import *
from .utilities import ifnone


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
    dtype, device = anchors.dtype, anchors.device
    dx = (
        torch.as_tensor(BBOX_REG_WEIGHTS[0], dtype=dtype, device=device)
        * (gt_x - x)
        / w
    )
    dy = (
        torch.as_tensor(BBOX_REG_WEIGHTS[1], dtype=dtype, device=device)
        * (gt_y - y)
        / h
    )
    dw = torch.as_tensor(BBOX_REG_WEIGHTS[2], dtype=dtype, device=device) * torch.log(
        gt_w / w
    )
    dh = torch.as_tensor(BBOX_REG_WEIGHTS[3], dtype=dtype, device=device) * torch.log(
        gt_h / h
    )

    targets = torch.cat((dx, dy, dw, dh), dim=1)
    return targets


def activ_2_bbox(
    activations: Tensor, anchors: Tensor, clip_activ: float = math.log(1000.0 / 16)
) -> Tensor:
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
    pred_boxes1 = (
        pred_ctr_x
        - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
    )
    pred_boxes2 = (
        pred_ctr_y
        - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
    )
    pred_boxes3 = (
        pred_ctr_x
        + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
    )
    pred_boxes4 = (
        pred_ctr_y
        + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
    )

    # Stack the co-ordinates to the bbox co-ordinates.
    pred_boxes = torch.stack(
        (pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2
    ).flatten(1)
    return pred_boxes


def matcher(
    targets: Tensor, anchors: Tensor, match_thr: float = None, back_thr: float = None
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
    matches = anchors.new(anchors.size(0)).zero_().long() - IGNORE_IDX
    match_thr = ifnone(match_thr, IOU_THRESHOLDS_FOREGROUND)
    back_thr = ifnone(back_thr, IOU_THRESHOLDS_BACKGROUND)
    # Calculate IOU between given targets & anchors
    iou_vals = box_iou(targets, anchors)
    # Grab the best ground_truth overlap
    vals, idxs = iou_vals.max(dim=0)
    # Grab the idxs
    matches[vals < back_thr] = BACKGROUND_IDX
    matches[vals > match_thr] = idxs[vals > match_thr]

    return matches
