import math
from typing import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.functional import Tensor
from .config import *
from .losses import focal_loss, smooth_l1_loss
from .utils import bbox_2_activ


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
        # Initialize with `kaiming_uniform`
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inps: List[Tensor]) -> List[Tensor]:
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


class RetinaNetHead(nn.Module):
    """
    A regression & classification head for use in `RetinaNet`

    Arguments :
    ---------
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
        out_channels: int,
        num_anchors: int,
        num_classes: int,
        prior: float,
    ) -> None:
        super(RetinaNetHead, self).__init__()
        self.classification_head = RetinaNetClassificationHead(
            in_channels, out_channels, num_anchors, num_classes, prior
        )
        self.regression_head = RetinaNetRegressionHead(
            in_channels, out_channels, num_anchors
        )

    def compute_loss(self, targets, outputs, anchors, matched_idxs):
        output_dict = {
            "classification_loss": self.classification_head.compute_loss(
                targets, outputs, matched_idxs
            ),
            "bbox_regression": self.regression_head.compute_loss(
                targets, outputs, anchors
            ),
        }
        return output_dict

    def forward(self, xb):
        output_dict = {
            "cls_preds": self.classification_head(xb),
            "bbox_preds": self.regression_head(xb),
        }
        return output_dict


class RetinaNetClassificationHead(nn.Module):
    """
    Classification Head for use in RetinaNet.
    This subnet  applies 4 3x3 conv layers, each with `out_channels`
    no of filters followed by a ReLU activation, followed by a 3x3
    conv layer with (num_classes*num_anchors) filters follwed by a 
    `sigmoid` prediction.

    Arguments:
    ---------
        in_channels  (int) : number of input channels.
        num_classes  (int) : total number of classes.
        num_anchors  (int) : no. of anchors per-spatial location.
        out_channels (out) : no. of channels for each conv_layer.
        prior (float)      : prior for `focal loss`.

    Returns:
    -------
        Tensor of shape [None, (height * width * num_anchors), num_classes] 
        where each item correspond to the binary predictions 
        per spatial location.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels,
        num_anchors: int,
        num_classes: int,
        prior: float,
    ) -> None:
        super(RetinaNetClassificationHead).__init__()
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
        self.class_subnet_output = nn.Conv2d(
            out_channels, num_anchors * num_classes, 3, stride=1, padding=1
        )
        # Initialize the Final Layer as given in :paper: `RetinaNet`
        torch.nn.init.normal_(self.class_subnet_output.weight, std=0.01)
        torch.nn.init.constant_(
            self.class_subnet_output.bias, -math.log((1 - prior) / prior)
        )

    @staticmethod
    def classification_loss(
        targets: List[Dict[str, Tensor]],
        outputs: Dict[str, Tensor],
        matched_idxs: List[Tensor],
    ):
        # ---------------------------------------------------------------
        # Calculate Classification Loss
        # ---------------------------------------------------------------
        loss = torch.tensor(0.0)
        classification_loss = torch.tensor(0.0)
        cls_preds = outputs["cls_preds"]

        for tgt, cls_pred, m_idx in zip(targets, cls_preds, matched_idxs):
            # no matched_idxs means there were no annotations in this image
            if m_idx.numel() == 0:
                gt_targs = torch.zeros_like(cls_pred)
                valid_idxs = torch.arange(cls_pred.shape[0])
                num_foreground = torch.tensor(0.0)
            else:
                # determine only the foreground
                foreground_idxs_ = m_idx >= 0
                num_foreground = foreground_idxs_.sum()
                gt_targs = torch.zeros_like(cls_pred)

                # create the target classification
                gt_targs[
                    foreground_idxs_, tgt["labels"][m_idx[foreground_idxs_]]
                ] = torch.tensor(1.0)

                # find indices for which anchors should be ignored
                valid_idxs = m_idx != IGNORE_IDX

            # compute the classification loss
            classification_loss += focal_loss(
                cls_pred[valid_idxs], gt_targs[valid_idxs], reduction="sum"
            ) / max(1, num_foreground)

        return classification_loss / len(targets)

    def forward(self, feature_maps):
        cls_preds = []

        for features in feature_maps:
            # in: [num_batches, ..., height, width]
            x = self.class_subnet(features)
            x = self.class_subnet_output(x)
            x = torch.sigmoid(x)
            # out: [num_batches, (num_anchors * num_classes), height, width ]
            N, _, H, W = x.shape
            x = x.view(N, -1, self.num_classes, H, W)
            x = x.permute(0, 3, 4, 1, 2)
            x = x.reshape(N, -1, self.num_classes)
            # out: [num_batches, (height*width*num_anchors), num_classes]
            cls_preds.append(x)

        # Concatenate along (height*wdth*num_anchors) dimension
        cls_preds = torch.cat(cls_preds, dim=1)
        return cls_preds


class RetinaNetRegressionHead(nn.Module):
    """
    Box subnet for regeressing from anchor boxes to ground truth labels.
    This subnet  applies 4 3x3 conv layers, each with `out_channels`
    no of filters followed by a ReLU activation, followed by a 3x3
    conv layer with (4 * num_anchors). For each anchor these 4 outputs, 
    predict the relative offset between the abhor box & ground_truth.

    Arguments:
    ----------
        in_channels (int) : number of input channels.
        out_channels (out): no. of channels for each conv_layer.
        num_anchors  (int): no. of anchors per-spatial location.

    Returns:
    --------
        Tensor of shape [None, (height * width * num_anchors), 4] where
        each item correspond to the relative offset between the anchor box & ground_truth per spatial location.
    """

    def __init__(self, in_channels: int, out_channels: int, num_anchors: int) -> None:
        super(RetinaNetRegressionHead, self).__init__()
        self.num_anchors = num_anchors
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
        self.box_subnet_output = nn.Conv2d(
            out_channels, num_anchors * 4, 3, padding=1, stride=1
        )
        # Initialize the Final Layer as given in :paper: `RetinaNet`
        torch.nn.init.normal_(self.output.weight, std=0.01)
        torch.nn.init.zeros_(self.output.bias)
        for layer in self.box_subnet.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

    @staticmethod
    def compute_loss(
        targets: List[Dict[str, Tensor]],
        outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ):
        # ---------------------------------------------------------------
        # Calculate Regression Loss
        # ---------------------------------------------------------------
        loss = torch.tensor(0.0)
        bbox_preds = outputs["bbox_regression"]

        for tgt, bbox, anc, idxs in zip(targets, bbox_preds, anchors, matched_idxs):
            # no matched_idxs means there were no annotations in this image
            if idxs.numel() == 0:
                continue

            matched_gts = tgt["boxes"][idxs.clamp(min=0)]
            # determine only the foreground indices, ignore the rest
            foreground_idxs_ = idxs >= 0
            num_foreground = foreground_idxs_.sum()

            # select only the foreground boxes
            matched_gts = matched_gts[foreground_idxs_, :]
            bbox = bbox[foreground_idxs_, :]
            anc = anc[foreground_idxs_, :]

            # compute the regression targets
            targs = bbox_2_activ(matched_gts, anc)

            # Compute loss
            loss += smooth_l1_loss(bbox, targs, reduction="sum") / max(
                1, num_foreground
            )

        return loss / max(1, len(targets))

    def forward(self, feature_maps):
        outputs = []

        for features in feature_maps:
            # in: [num_batches, ..., height, width]
            x = self.box_subnet(features)
            x = self.box_subnet_output(x)
            # out: [num_batches, (num_anchors * num_classes), height, width ]
            N, _, H, W = x.shape
            x = x.view(N, -1, 4, H, W)
            x = x.permute(0, 3, 4, 1, 2)
            x = x.reshape(N, -1, 4)
            # out: [num_batches, (height*width*num_anchors), num_classes]
            outputs.append(x)
        # Concatenate along (height*wdth*num_anchors) dimension
        outputs = torch.cat(outputs, dim=1)
        return outputs
