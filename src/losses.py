from typing import *

import torch
import torch.nn.functional as F
from torch import nn
from torch.functional import Tensor

from .config import *
from .utils import bbox_2_activ, matcher


class RetinaNetLosses(nn.Module):
    def __init__(self, num_classes) -> None:
        super(RetinaNetLosses, self).__init__()
        self.n_c = num_classes
        self.alpha = FOCAL_LOSS_ALPHA
        self.gamma = FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = SMOOTH_L1_LOSS_BETA

        # maintain an EMA of #foreground tostabilize the normalizer.
        self.loss_normalizer = 100
        self.loss_normalizer_momentum = 0.9

    def focal_loss(self, clas_pred: Tensor, clas_tgt: Tensor) -> Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            1. inputs: A float tensor of arbitrary shape.
                       The predictions for each example.
            2. targets: A float tensor with the same shape as inputs. 
                        Stores the binary classification label for each element in inputs
                        (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor
        """
        ps = torch.sigmoid(clas_pred.detach())
        weights = clas_tgt * (1 - ps) + (1 - clas_tgt) * ps
        alphas = (1 - clas_tgt) * self.alpha + clas_tgt * (1 - self.alpha)
        weights.pow_(self.gamma).mul_(alphas)
        clas_loss = F.binary_cross_entropy_with_logits(
            clas_pred, clas_tgt, weights, reduction="sum"
        )
        return clas_loss

    def smooth_l1_loss(self, input: Tensor, target: Tensor):
        if self.smooth_l1_loss_beta < 1e-5:
            loss = torch.abs(input - target)
        else:
            n = torch.abs(input - target)
            cond = n < self.smooth_l1_loss_beta
            loss = torch.where(
                cond,
                0.5 * n ** 2 / self.smooth_l1_loss_beta,
                n - 0.5 * self.smooth_l1_loss,
            )
        return loss.sum()

    def calc_loss(
        self,
        anchors: Tensor,
        clas_pred: Tensor,
        bbox_pred: Tensor,
        clas_tgt: Tensor,
        bbox_tgt: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate loss for class & box subnet of retinanet.
        """
        # Match boxes with anchors to get `background`, `ignore` and `foreground` positions
        matches = matcher(anchors, bbox_tgt)

        # create filtering mask to filter `background` and `ignore` classes from the bboxes
        bbox_mask = matches >= 0

        if bbox_mask.sum() != 0:
            gt_anchor_deltas = bbox_2_activ(bbox_tgt, anchors)
            clas_pred = clas_pred[bbox_mask]
            gt_anchor_deltas = gt_anchor_deltas[bbox_mask]
            # regression loss
            bb_loss = self.smooth_l1_loss(bbox_pred, gt_anchor_deltas)
        else:
            bb_loss = 0.0

        # filtering mask to filter `ignore` classes from the class predicitons
        matches.add_(1)
        clas_mask = matches >= 0
        clas_pred = clas_pred[clas_mask]

        clas_tgt = clas_tgt + 1

        # Add background class to account for background in `matches`. When there are no
        # matches
        clas_tgt = torch.cat([clas_tgt.new_zeros(1).long(), clas_tgt])
        clas_tgt = clas_tgt[matches[clas_mask]]

        # no loss for the first(background) class
        clas_tgt = F.one_hot(clas_tgt, num_classes=self.n_c + 1)[:, 1:].to(
            clas_pred.dtype
        )

        # classification loss
        clas_loss = self.focal_loss(clas_pred, clas_tgt) / torch.clamp(
            bbox_mask.sum(), min=1.0
        )

        # Normalize Loss
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        )
        return clas_loss.div_(self.loss_normalizer), bb_loss.div_(self.loss_normalizer)

    def forward(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: List[Tensor],
        anchors: List[Tensor],
    ):
        # extract the class_predictions & bbox_predictions from the RetinaNet Head Outputs
        clas_preds, bbox_preds = head_outputs["cls_preds"], head_outputs["bbox_preds"]
        losses = {}
        losses["classification_loss"] = []
        losses["regression_loss"] = []
        # Total number of Images
        num_ims = len(bbox_preds)

        for cls_pred, bb_pred, targs, ancs in zip(
            clas_preds, bbox_preds, targets, anchors
        ):

            # Extract the Labels & boxes from the targets
            class_targs, bbox_targs = targs["labels"], targs["boxes"]
            # Compute loss
            clas_loss, bb_loss = self.calc_loss(
                ancs, cls_pred, bb_pred, class_targs, bbox_targs
            )
            # Append Losses
            losses["classification_loss"].append(clas_loss)
            losses["regression_loss"].append(bb_loss)

        # Normalize losses
        losses["classification_loss"] = sum(losses["classification_loss"]) / num_ims

        losses["regression_loss"] = sum(losses["regression_loss"]) / num_ims

        return losses
