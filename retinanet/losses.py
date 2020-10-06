from typing import *

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .box_utils import bbox_2_activ, matcher
from .config import *


class RetinaNetLosses(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(RetinaNetLosses, self).__init__()
        self.n_c   = num_classes
        self.alpha = FOCAL_LOSS_ALPHA
        self.gamma = FOCAL_LOSS_GAMMA
        self.beta  = SMOOTH_L1_LOSS_BETA

    def smooth_l1_loss(self, input: Tensor, target: Tensor) -> torch.Tensor:
        """Computes SmoothL1Loss"""
        if self.beta < 1e-5:
            loss = torch.abs(input - target)
        else:
            n = torch.abs(input - target)
            cond = n < self.beta
            loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        return loss.sum()

    def focal_loss(self, clas_pred: Tensor, clas_tgt: Tensor) -> Tensor:
        """
        Focal Loss used in RetinaNet: https://arxiv.org/abs/1708.02002.
        
        Args:
            1. clas_pred: A float tensor of arbitrary shape.
                         The predictions for each example.
            2. clas_tgt: A float tensor with the same shape as inputs.
                         Stores the binary classification label for each element in inputs
                        (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor
        """
        ps = torch.sigmoid(clas_pred.detach())
        weights = clas_tgt * (1 - ps) + (1 - clas_tgt) * ps
        alphas = (1 - clas_tgt) * self.alpha + clas_tgt * (1 - self.alpha)
        weights.pow_(self.gamma).mul_(alphas)
        clas_loss = F.binary_cross_entropy_with_logits(clas_pred, clas_tgt, weights, reduction="sum")
        return clas_loss

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
        # This is also number of foreground
        bbox_mask = matches >= 0

        if bbox_mask.sum() != 0:
            bbox_pred = bbox_pred[bbox_mask]
            bbox_tgt = bbox_tgt[matches[bbox_mask]]
            # match the targets with anchors to get the bboxes
            bbox_tgt = bbox_2_activ(bbox_tgt, anchors[bbox_mask])
            bb_loss = self.smooth_l1_loss(bbox_pred, bbox_tgt)
        else:
            bb_loss = torch.tensor(0.0).to(bbox_pred.device)

        # filtering mask to filter `ignore` classes from the class predicitons
        matches.add_(1)
        clas_mask = matches >= 0
        clas_pred = clas_pred[clas_mask]

        # model is going to predict classes which are going to be in the range of [0, num_classes]
        # 0 is reserved for the background class for which no loss is calculate , so 
        # we will add 1 to all the class_predictions to shift the predicitons range from
        # [0, num_classes) -> [1, num_classes]
        clas_pred = clas_pred + 1

        # # clas_tgt : [0, num_classes) -> [1, num_classes]
        # clas_tgt = clas_tgt + 1
        # # no need to add +1 since clas_tgt: [1, num_classes]
        # Add background class to account for background in `matches`.

        # When there are no matches
        # bg class is predicted so, we need to add the backgorund class
        # to each of the targets , or else there will be index error, as
        # bboxes are predicted for background class as well
        # add 0 label for the background class
        clas_tgt = torch.cat([clas_tgt.new_zeros(1).long(), clas_tgt])
        clas_tgt = clas_tgt[matches[clas_mask]]

        # convert the integer lables into a one-hot vector and omit
        # the first column which corresponds to the 0th class as ,
        # no loss for the first(background) class
        clas_tgt = F.one_hot(clas_tgt, num_classes=self.n_c + 1)[:, 1:]
        clas_tgt = clas_tgt.to(clas_pred.dtype)
        # classification loss
        clas_loss = self.focal_loss(clas_pred, clas_tgt)

        # Normalize Loss with num foregrounds
        bb_loss.to(clas_loss.dtype).div_(torch.clamp(bbox_mask.sum(), min=1.0))
        clas_loss.div_(torch.clamp(bbox_mask.sum(), min=1.0))

        return bb_loss, clas_loss

    def forward(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: List[Tensor],
        anchors: List[Tensor],
    ) -> Dict[str, Tensor]:
        # extract the class_predictions & bbox_predictions from the RetinaNet Head Outputs
        clas_preds, bbox_preds = head_outputs["cls_preds"], head_outputs["bbox_preds"]
        losses = {}
        # List to store losses
        classification_losses = []
        regression_losses = []

        for cls_pred, bb_pred, targs, ancs in zip(clas_preds, bbox_preds, targets, anchors):
            # Extract the Labels & boxes from the targets
            class_targs, bbox_targs = targs["labels"], targs["boxes"]
            # Compute losses
            model_losses = self.calc_loss(ancs,cls_pred, bb_pred, class_targs, bbox_targs)
            # unpack the losses                                                     
            regression_loss, classification_loss = model_losses
            # Append Losses of all the batches
            classification_losses.append(classification_loss)
            regression_losses.append(regression_loss)

        # Average  the losses
        classification_losses = sum(classification_losses) / len(targets)

        regression_losses = sum(regression_losses) / len(targets)

        # Add losses to dictionary
        losses["classification_loss"] = classification_losses
        losses["regression_loss"] = regression_losses
        return losses
