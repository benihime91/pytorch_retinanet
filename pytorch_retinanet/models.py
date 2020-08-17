from typing import List
from torch.functional import Tensor
import torch.nn.functional as F
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
        # `conv layers` to calculate `p3`
        self.conv_c3_1x1 = nn.Conv2d(C_3_size, out_channels, 1, 1, padding=0)
        self.conv_c3_3x3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        # `conv layers` to calculate `p4`
        self.conv_c4_1x1 = nn.Conv2d(C_4_size, out_channels, 1, 1, padding=0)
        self.conv_c4_3x3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        # `conv layers` to calculate `p5`
        self.conv_c5_1x1 = nn.Conv2d(C_5_size, out_channels, 1, 1, padding=0)
        self.conv_c5_3x3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        # `conv layers` to calculate `p6`
        self.conv_c6_3x3 = nn.Conv2d(C_5_size, out_channels, 3, stride=2, padding=1)
        # `conv layers` to calculate `p7`
        self.conv_c7_3x3 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
        # `upsample layer` to increase `output_size` for `elementwise-additions`
        # with previous pyramid level
        self.upsample_2x = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inps:List[Tensor]):
        C3, C4, C5 = inps
        # Calculate outputs from C3, C4, C5 [feature maps] using 
        # `1x1 stride stide 1 convs` on `C3`, `C4`, `C5`
        p3_output  = self.conv_c3_1x1(C3)
        p4_output  = self.conv_c4_1x1(C4)
        p5_output  = self.conv_c4_1x1(C5)

        # Upsample & add ouputs[element-wise]: (p4 and p5) & (p3 and p4)
        p4_output  = p4_output + self.upsample_2x(p5_output)
        p3_output  = p3_output + self.upsample_2x(p4_output)
        # `3x3 stride-1 Convs` to obtain `p3`, `p4`, `p5`
        p3_output  = self.conv_c3_3x3(p3_output)
        p4_output  = self.conv_c4_3x3(p4_output)
        p5_output  = self.conv_c5_3x3(p5_output)

        # Calculate p6 & p7
        # `p6` is obtained via a `3x3 stride-2 conv` on `C5`
        p6_output  = self.conv_c6_3x3(C5)
        # `p7` is computed by applying `ReLU` followed a `3x3 stride-2 conv` on `p6`
        p7_output  = self.conv_c7_3x3(F.relu(p6_output))
        return [p3_output, p4_output, p5_output, p6_output, p7_output]

# if __name__ == '__main__':
#     a, b, c = [1, 2, 3]
#     print(a)
#     print(b)
#     print(c)
