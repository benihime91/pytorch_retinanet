import math
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import nn as nn
from torch.functional import Tensor

from ..config import *
from .utils import retinanet_loss

__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

funcs = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,
}


class EmptyLayer(nn.Module):
    " PlaceHolder for `AvgPool` and `FC Layer` "

    def __init__(self) -> None:
        super(EmptyLayer, self).__init__()

    def forward(self, xb):
        return xb


# Dictionary to store Intermediate Outputs
inter_outs = {}


class BackBone(nn.Module):
    def __init__(
        self,
        kind: str = "resnet18",
        hook_fn: Callable = None,
        pretrained: bool = True,
        freeze_bn: bool = True,
    ) -> None:
        """Create a Backbone from `kind`"""
        super(BackBone, self).__init__()
        build_fn = funcs[kind]
        self.backbone = build_fn(pretrained=pretrained)
        self.backbone.avgpool = EmptyLayer()
        self.backbone.fc = EmptyLayer()

        # Get the Feature maps from the intermediate layer outputs
        self.backbone.layer2.register_forward_hook(hook_fn)
        self.backbone.layer3.register_forward_hook(hook_fn)
        self.backbone.layer4.register_forward_hook(hook_fn)

        # Freeze batch_norm: Not sure why ?? but every other implementation does it
        if freeze_bn:
            for layer in self.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()

    def forward(self, xb: Tensor) -> List[Tensor]:
        _ = self.backbone(xb)
        out = [
            inter_outs[self.backbone.layer2],
            inter_outs[self.backbone.layer3],
            inter_outs[self.backbone.layer4],
        ]
        return out


def get_backbone(
    kind: str = "resnet18", pretrained: bool = True, freeze_bn: bool = True
) -> nn.Module:
    """
    Returns a `ResNet` Backbone.

    Args:
        1. kind       : (str) name of the resnet model eg: `resnet18`.
        2. pretrained : (bool) wether to load pretrained `imagenet` weights.
        3. freeze_bn  : (bool) wether to freeze `BatchNorm` layers.

    Example:
        >>> m = get_backbone(kind='resnet18')
    """
    assert kind in __all__, f"`kind` must be one of {__all__} got {kind}"

    def hook_outputs(self, inp, out) -> None:
        # Function to Hook Intermediate Outputs
        inter_outs[self] = out

    backbone = BackBone(
        kind=kind, hook_fn=hook_outputs, pretrained=pretrained, freeze_bn=freeze_bn
    )

    return backbone


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
        self.upsample_2x = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, inps: List[Tensor]):
        C3, C4, C5 = inps
        # Calculate outputs from C3, C4, C5 [feature maps] using
        # `1x1 stride stide 1 convs` on `C3`, `C4`, `C5`
        p3_output = self.conv_c3_1x1(C3)
        p4_output = self.conv_c4_1x1(C4)
        p5_output = self.conv_c5_1x1(C5)

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

    def __init__(
        self, in_channels: int, out_channels: int = 256, num_anchors: int = 9
    ) -> None:
        super(BoxSubnet, self).__init__()
        # Successive conv_layers
        self.box_subnet = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(out_channels, num_anchors * 4, 3, padding=1, stride=1)
        # out shape: [batch_size, (num_anchors * 4), height, width]

        torch.nn.init.normal_(self.output.weight, std=0.01)
        torch.nn.init.zeros_(self.output.bias)

        for layer in self.box_subnet.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.num_anchors = num_anchors

    def forward(self, xb: List[Tensor]) -> Tensor:
        outputs = []

        for features in xb:
            x = self.box_subnet(features)
            x = self.output(x)

            # Reshape output from :
            # (batch_size, 4 * num_anchors, H, W) -> (batch_size, H*W*num_anchors, 4).
            N, _, H, W = x.shape
            x = x.view(N, -1, 4, H, W)
            x = x.permute(0, 3, 4, 1, 2)
            x = x.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            outputs.append(x)

        return torch.cat(outputs, dim=1)


class ClassSubnet(nn.Module):
    """
    Class subnet for classifying anchor boxes.
    This subnet  applies 4 3x3 conv layers, each with `out_channels`
    no of filters followed by a ReLU activation, followed by a 3x3
    conv layer with (num_classes*num_anchors) filters follwed by a 
    `sigmoid` prediction.

    Args:
        in_channels  (int) : number of input channels.
        num_classes  (int) : total number of classes.
        num_anchors  (int) : no. of anchors per-spatial location.
        prior (float)      : prior for `focal loss`.
        out_channels (out) : no. of channels for each conv_layer.

    Returns:
        Tensor of shape [None, (height * width * num_anchors), num_classes] 
        where each item correspond to the binary predictions per spatial location.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int,
        prior: float = PRIOR,
        out_channels: int = 256,
    ) -> None:

        super(ClassSubnet, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.class_subnet = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(out_channels, (num_anchors * num_classes), 3, padding=1)

        for layer in self.class_subnet.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.normal_(self.output.weight, std=0.01)
        torch.nn.init.constant_(self.output.bias, -math.log((1 - prior) / prior))

    def forward(self, xb: List[Tensor]) -> Tensor:
        outputs = []

        for feature in xb:
            x = self.class_subnet(feature)
            # `F.sigmoid` is `deprecated`
            x = torch.sigmoid(self.output(x))
            # Permute classification output from :
            # (batch_size, num_anchors * num_classes, H, W) to (batch_size, H * W * num_anchors, num_classes).
            N, _, H, W = x.shape
            x = x.view(N, -1, self.num_classes, H, W).permute(0, 3, 4, 1, 2)
            x = x.reshape(N, -1, self.num_classes)

            outputs.append(x)

        outputs = torch.cat(outputs, dim=1)
        return outputs


class RetinaNetHead(nn.Module):
    """
    A Regression & Classification Head for use in RetinaNet.

    Args:
        in_channels(int)  : number of input channels.
        out_channels (int): number of output feature channels.
        num_anchors (int) : number of anchors per_spatial_location.
        num_classes  (int): number of classes to be predicted.
        prior      (float): value of `p estimated` by the model for the rare class (foreground) at the 
                            start of training.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        out_channels: int = 256,
        num_anchors: int = 9,
        prior: float = PRIOR,
    ) -> None:

        super(RetinaNetHead, self).__init__()

        # Initialize `ClassSubnet`
        self.classification_head = ClassSubnet(
            in_channels, num_classes, num_anchors, prior, out_channels
        )

        # Initialize `BoxSubnet`
        self.regression_head = BoxSubnet(in_channels, out_channels, num_anchors)

    def forward(self, xb: List[Tensor]) -> Dict[str, Tensor]:
        cls_logits = self.classification_head(xb)
        bbox_regressions = self.regression_head(xb)

        return {"logits": cls_logits, "bboxes": bbox_regressions}

    @staticmethod
    def retinanet_focal_loss(
        targets: List[Dict[str, Tensor]],
        ouptuts: Dict[str, Tensor],
        anchors: List[Tensor],
    ):

        loss = retinanet_loss(targets, ouptuts, anchors)
        return loss
