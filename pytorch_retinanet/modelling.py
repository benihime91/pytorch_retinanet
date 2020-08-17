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

__small__ = ['resnet18', 'resnet34']
__big__ = ['resnet50', 'resnet101', 'resnet101', 'resnet152']


class FPN(nn.Module):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, C_3_size, C_4_size, C_5_size, out_channels=256) -> None:
        super(FPN, self).__init__()
        # `conv layers` to calculate `p3`
        self.conv_c3_1x1 = nn.Conv2d(C_3_size, out_channels, 1, 1, padding=0)
        self.conv_c3_3x3 = nn.Conv2d(
            out_channels, out_channels, 3, 1, padding=1)
        # `conv layers` to calculate `p4`
        self.conv_c4_1x1 = nn.Conv2d(C_4_size, out_channels, 1, 1, padding=0)
        self.conv_c4_3x3 = nn.Conv2d(
            out_channels, out_channels, 3, 1, padding=1)
        # `conv layers` to calculate `p5`
        self.conv_c5_1x1 = nn.Conv2d(C_5_size, out_channels, 1, 1, padding=0)
        self.conv_c5_3x3 = nn.Conv2d(
            out_channels, out_channels, 3, 1, padding=1)
        # `conv layers` to calculate `p6`
        self.conv_c6_3x3 = nn.Conv2d(
            C_5_size, out_channels, 3, stride=2, padding=1)
        # `conv layers` to calculate `p7`
        self.conv_c7_3x3 = nn.Conv2d(
            out_channels, out_channels, 3, stride=2, padding=1)

        # `upsample layer` to increase `output_size` for `elementwise-additions`
        # with previous pyramid level
        self.upsample_2x = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inps: List[Tensor]):
        C3, C4, C5 = inps
        # Calculate outputs from C3, C4, C5 [feature maps] using
        # `1x1 stride stide 1 convs` on `C3`, `C4`, `C5`
        p3_output = self.conv_c3_1x1(C3)
        p4_output = self.conv_c4_1x1(C4)
        p5_output = self.conv_c4_1x1(C5)

        # Upsample & add ouputs[element-wise]: (p4 and p5) & (p3 and p4)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        # `3x3 stride-1 Convs` to obtain `p3`, `p4`, `p5`
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)

        # Calculate p6 & p7
        # `p6` is obtained via a `3x3 stride-2 conv` on `C5`
        p6_output = self.conv_c6_3x3(C5)
        # `p7` is computed by applying `ReLU` followed a `3x3 stride-2 conv` on `p6`
        p7_output = self.conv_c7_3x3(F.relu(p6_output))
        return [p3_output, p4_output, p5_output, p6_output, p7_output]


class BoxSubnet(nn.Module):
    """
    Box subnet for regeressing from anchor boxes to ground truth labels.
    This subnet  applies 4 3x3 conv layers, each with `out_channels`
    no of filters followed by a ReLU activation, followed by a 3x3
    conv layer with (4 * num_anchors). For each anchor these 4 outputs, 
    predict the relative offset between the abhor box & ground_truth.

    Args:
        in_channels (int) : number of input channels.
        out_channels (out): no. of channels for each conv_layer.
        num_anchors  (int): no. of anchors per-spatial location.

    Returns:
        Tensor of shape [None, (height * width * num_anchors), 4] where
        each item correspond to the relative offset between the anchor box & ground_truth
        per spatial location.
    """

    def __init__(self, in_channels: int, out_channels: int = 256, num_anchors: int = 9) -> None:
        super(BoxSubnet, self).__init__()
        # Successive conv_layers
        self.box_subnet(
            nn.Conv2d(in_channels, out_channels,  3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(out_channels, num_anchors * 4, 3, padding=1)
        # out shape: [batch_size, (num_anchors * 4), height, width]

    def forward(self, xb) -> Tensor:
        x = self.box_subnet(xb)
        x = self.output(x)
        # Reshape output from :
        # [batch_size, (num_anchors * 4), height, width] -> [batch_size, height, width, (num_anchors*4)]
        x = x.permute(0, 2, 3, 1)
        # Flatten output into shape: [batch_size, (height * width * num_anchors), 4]
        # height * width * num_anchors =  Total number of anchors in the Feature Map.
        return x.contiguous().view(x.shape[0], -1, 4)


class ClassSubnet(nn.Module):
    """
    Class subnet for classifying anchor boxes.
    This subnet  applies 4 3x3 conv layers, each with `out_channels`
    no of filters followed by a ReLU activation, followed by a 3x3
    conv layer with (num_classes*num_anchors) filters follwed by a 
    `sigmoid` prediction.

    Args:
        in_channels (int) : number of input channels.
        num_classes  (int): total number of classes.
        num_anchors  (int): no. of anchors per-spatial location.
        prior (float)     : prior for `focal loss`.
        out_channels (out): no. of channels for each conv_layer.

    Returns:
        Tensor of shape [None, (height * width * num_anchors), num_classes] 
        where each item correspond to the binary predictions per spatial location.
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 num_anchors: int = 9,
                 prior: float = 0.01,
                 out_channels: int = 256) -> None:
        super(ClassSubnet, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.class_subnet = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,  3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(
            out_channels, (num_anchors * num_classes), 3, padding=1)

    def forward(self, xb) -> Tensor:
        x = self.class_subnet(xb)
        x = F.sigmoid(self.output(x))
        # out : [batch_size, (num_anchors * num_classes), height, width]
        x = x.permute(0, 2, 3, 1)
        # out: [batch_size, height, width, (num_anchors * num_classes)]
        batch_size, height, width, _ = x.shape
        x = x.view(batch_size, height, width,
                   self.num_anchors, self.num_classes)
        x = x.contiguous().view(x.shape[0], -1, self.num_classes)
        return x


class Retinanet(nn.Module):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    def __init__(self,
                 num_classes: int,
                 backbone_kind: str = 'resnet18',
                 num_anchors: int = 9,
                 prior: float = 0.01,
                 device: str = 'cpu',
                 pretrained: bool = True) -> None:

        assert backbone_kind in __small__ + \
            __big__, f" Expected `backbone_kind` to be one of {__small__+__big__} got {backbone_kind}"

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

        self.class_subnet = ClassSubnet(in_channels=256,
                                        num_classes=num_classes,
                                        num_anchors=num_anchors,
                                        prior=prior,
                                        out_channels=256)

        self.box_subnet = BoxSubnet(in_channels=256, out_channels=256)
        self.anchor_generator = AnchorGenerator(device=torch.device(device))
        self.activ2bbox = Activ2BoxTransform(device=torch.device(device))
        self.clip_boxes = ClipBoxes()
        self.focal_loss = None
