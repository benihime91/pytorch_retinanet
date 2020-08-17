import torch
import torch.nn as nn
from typing import *
from .anchors import ifnone
import numpy as np
import math


class Activ2BoxTransform(nn.Module):
    # For each anchor, we have one class predicted by the classifier and 4 floats `p_y, p_x, p_h, p_w` predicted by the regressor.
    # If the corresponding anchor as a center in `anc_y`, `anc_x` with dimensions `anc_h`, `anc_w`,
    # the predicted bounding box has those characteristics:
    # >>> center = [p_y * anc_h + anc_y, p_x * anc_w + anc_x]
    # >>> height = anc_h * exp(p_h)
    # >>> width = anc_w * exp(p_w)
    # The idea is that a prediction of `(0, 0, 0, 0)` corresponds to the anchor itself.
    # The next function converts the activations of the model in bounding boxes.
    def __init__(self, scales: List[float], device: torch.device = torch.device('cpu')) -> None:
        super(Activ2BoxTransform, self).__init__()
        scales = np.array(ifnone(scales, [0.1, 0.1, 0.2, 0.2]))
        self.scales = torch.from_numpy(scales).float().to(device)
        self._device = device

    @property
    def device(self):
        return self._device


    def forward(self, anchors, activations):
        """convert activations of the model into bounding boxes."""

        # Extract w,h,x_center,y_center from x1,y1,x2,y2 anchors.
        widths  = anchors[:, :, 2] - anchors[:, :, 0]
        heights = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x   = anchors[:, :, 0] - 0.5 * widths
        ctr_y   = anchors[:, :, 1] - 0.5 * heights

        # Get box regression transformation deltas(dx, dy, dw, dh) that can be used
        # to transform the `activations` into the `pred_boxes`.
        dx = activations[:, :, 0] * self.scales[0]
        dy = activations[:, :, 1] * self.scales[1]
        dw = activations[:, :, 2] * self.scales[2]
        dh = activations[:, :, 3] * self.scales[3]

        # Extrapolate bounding boxes on anchors from the model activations.
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        # Convert to x1y1x2y2 format
        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)
        return pred_boxes

    
class ClipBoxes(nn.Module):
    '''
    Clip the `Height` & `Width` of the boxes to 
    the `Height` and `Width` of the Image.
    '''
    def __init__(self) -> None:
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
        return boxes
