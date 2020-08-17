import torch.nn as nn
import torch
import math
from torchvision.ops import nms
from .backbone import get_backbone
from .anchors import AnchorGenerator
from .utils import Activ2BoxTransform, ClipBoxes


class FPN(nn.Module):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, C_3_size, C_4_size, C_5_size, out_channels=256) -> None:
        super(FPN, self).__init__()

        
        self.conv_P5_1x1  = nn.Conv2d(C_5_size, out_channels, 1, 1, padding=0)
        self.conv_P5_3x3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

        self.conv_P4_1x1 = nn.Conv2d(C_4_size, out_channels, 1, 1, padding=0)
        self.conv_P4_3x3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

        self.conv_P3_1x1 = nn.Conv2d(C_3_size, out_channels, 1, 1, padding=0)
        self.conv_P3_3x3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

        self.conv_P5_1x1 = nn.Conv2d(C_5_size, out_channels, 1, 1, padding=0)
        self.conv_P5_1x1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

