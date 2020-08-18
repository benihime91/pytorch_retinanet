import math
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from torchvision.ops.boxes import box_iou, nms, remove_small_boxes

from .anchors import AnchorGenerator
from .backbone import get_backbone
from .modelling_utils import FPN, RetinaNetHead
from .utils import EncoderDecoder

__small__ = ['resnet18', 'resnet34']
__big__ = ['resnet50', 'resnet101', 'resnet101', 'resnet152']


class Retinanet(nn.Module):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    def __init__(self,
                 num_classes: int,
                 backbone_kind: str = 'resnet18',
                 num_anchors: int = 9,
                 prior: float = 0.01,
                 device: torch.device = torch.device('cpu'),
                 pretrained: bool = True) -> None:

        assert backbone_kind in __small__ + \
            __big__, f" Expected `backbone_kind` to be one of {__small__+__big__} got {backbone_kind}"

        self._device = device

        # Get the back bone of the Model
        self.backbone = get_backbone(backbone_kind, pretrained)

        # Grab the backbone output channels
        if backbone_kind in __small__:
            self.fpn_szs = [
                self.backbone.backbone.layer2[1].out_channels,
                self.backbone.backbone.layer3[1].out_channels,
                self.backbone.backbone.layer4[1].out_channels,
            ]
        elif backbone_kind in __big__:
            self.fpn_szs = [
                self.backbone.backbone.layer2[2].out_channels,
                self.backbone.backbone.layer3[2].out_channels,
                self.backbone.backbone.layer4[2].out_channels,
            ]

        # get the FPN
        self.fpn = FPN(self.fpn_szs[0], self.fpn_szs[1],
                       self.fpn_szs[2], out_channels=256)

        # Get RetinaNet Head
        self.retinanet_head = RetinaNetHead(
            256, num_classes, 256, num_anchors, prior)

    @property
    def device(self):
        return self._device

    def forward(self, batched_inputs):
        pass
