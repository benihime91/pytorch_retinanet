import math
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor


def bbox_2_activ(ground_truth_boxes: Tensor, anchors: Tensor) -> Tensor:
    """
    Convert `ground_truths` to match the model `activations` to calculate `loss`.
    """
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
    dx = (gt_x-x)/w
    dy = (gt_y-y)/h
    dw = torch.log(gt_w / w)
    dh = torch.log(gt_h / h)

    targets = torch.cat((dx, dy, dw, dh), dim=1)
    return targets


def activ_2_bbox(activations: Tensor, anchors: Tensor, clip_activ: float = math.log(1000. / 16)) -> Tensor:
    """Converts the `activations` of the `model` to bounding boxes."""

    if anchors.device != activations.device:
        anchors = anchors.to(activations.device)

    w = anchors[:, 2] - anchors[:, 0]
    h = anchors[:, 3] - anchors[:, 1]
    x = anchors[:, 0] + 0.5 * w
    y = anchors[:, 1] + 0.5 * h

    dx = activations[:, 0::4]
    dy = activations[:, 1::4]
    dw = activations[:, 2::4]
    dh = activations[:, 3::4]
    # Clip activations
    dw = torch.clamp(dw, max=clip_activ)
    dh = torch.clamp(dh, max=clip_activ)

    # Extrapolate bounding boxes on anchors from the model activations.
    pred_ctr_x = x[:, None] + dx * w[:, None]
    pred_ctr_y = y[:, None] + dy * h[:, None]
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]

    # Convert bbox shape from xywh to x1y1x2y2
    pred_boxes1 = pred_ctr_x - \
        torch.tensor(0.5, dtype=pred_ctr_x.dtype,
                     device=pred_w.device) * pred_w
    pred_boxes2 = pred_ctr_y - \
        torch.tensor(0.5, dtype=pred_ctr_y.dtype,
                     device=pred_h.device) * pred_h
    pred_boxes3 = pred_ctr_x + \
        torch.tensor(0.5, dtype=pred_ctr_x.dtype,
                     device=pred_w.device) * pred_w
    pred_boxes4 = pred_ctr_y + \
        torch.tensor(0.5, dtype=pred_ctr_y.dtype,
                     device=pred_h.device) * pred_h

    # Stack the co-ordinates to the bbox co-ordinates.
    pred_boxes = torch.stack(
        (pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)

    return pred_boxes


class EncoderDecoder:
    def encode(self, gt_bboxes: List[Tensor], anchors: List[Tensor]) -> List[Tensor]:
        "Return the target of the model on `anchors` for the `gt_bboxes`."
        boxes_per_image = [len(b) for b in gt_bboxes]
        # Create Tensor from Given Lists
        gt_bboxes = torch.cat(gt_bboxes, dim=0)
        anchors = torch.cat(anchors, dim=0)
        # computs targets of the model
        targets = bbox_2_activ(gt_bboxes, anchors)
        # convert to List
        targets = targets.split(boxes_per_image, 0)
        return targets

    def decode(self, activations: Tensor, anchors: List[Tensor]) -> Tensor:
        anchors_per_image = [a.size(0) for a in anchors]
        anchors = torch.cat(anchors, dim=0)

        dims = 0
        # Calculate Total size of anchors
        for dim in anchors_per_image:
            dims += dim

        pred_boxes = activ_2_bbox(activations.reshape(dims, -1), anchors)
        return pred_boxes.reshape(dims, -1, 4)


def smooth_l1_loss(inp, targs):
    """
    Computes `Smooth_L1_Loss`
    """
    return F.smooth_l1_loss(inp, targs, size_average=True)


# Set Values for IGNORE & BACKGROUND
IGNORE_IDX = -2
BACKGROUND_IDX = -1


class Matcher:
    """
    Match `anchors` to targets. -1 is match to background, -2 is ignore.

    From : https://github.com/fastai/course-v3/blob/9c83dfbf9b9415456c9801d299d86e099b36c86d/nbs/dl2/pascal.ipynb

    - for each anchor we take the maximum overlap possible with any of the targets.

    - if that maximum overlap is less than 0.4, we match the anchor box to background,
      the classifier's target will be that class.

    - if the maximum overlap is greater than 0.5, we match the anchor box to that ground truth object.
      The classifier's target will be the category of that target.

    - if the maximum overlap is between 0.4 and 0.5, we ignore that anchor in our loss computation.
    """

    def __init__(self, match_thr: float = 0.5, back_thr: float = 0.4) -> None:
        """
        Args:
            match_thr (float) : IOU values greater than or equal to this are `candidate values`.
            back_thr (float)  : IOU values grater less than this are assigned either `ignore` or `-1`.
        """
        assert match_thr > back_thr, "Threshold for `match` should be greater than `background`"
        self.match_thr = match_thr
        self.back_thr = back_thr

    def __call__(self, iou_vals: Tensor) -> Tensor:
        """
        Args:
            iou_vals(Tensor):  A MxN Tensor containing the IOU vals between M ground_truths & N predicted elements.
        """
        # Grab the best ground_truth overlap
        vals, idxs = iou_vals.max(dim=0)

        # Assign candidate matches with low quality to negative (unassigned) values
        # Threshold less than `back_thr` gets assigned -1 : background
        idxs[vals < self.back_thr] = torch.tensor(BACKGROUND_IDX)
        # Threshold between `match_thr` & `back_thr` gets assigned -2: ignore
        idxs[(vals >= self.back_thr) & (
            vals < self.match_thr)] = torch.tensor(IGNORE_IDX)
        return idxs


def focal_loss(
        inputs,
        targets,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none"):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .

    Args:
        inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = inputs
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
