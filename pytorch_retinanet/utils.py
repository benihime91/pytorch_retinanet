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
    Computes Smooth `Smooth_L1_Loss`
    """
    return F.smooth_l1_loss(inp, targs, size_average=True)
