from typing import *
import torch
import torch.nn.functional as F
from torch import nn
from torch.functional import Tensor
from .config import *
from .utils import bbox_2_activ


def focal_loss(inputs: Tensor, targets: Tensor,) -> Tensor:
    """
    Focal Loss
    """
    alpha = FOCAL_LOSS_ALPHA
    gamma = FOCAL_LOSS_GAMMA

    ps = torch.sigmoid(inputs.detach())
    weights = targets * (1 - ps) + (1 - targets) * ps
    alphas = (1 - targets) * alpha + targets * (1 - alpha)
    weights.pow_(gamma).mul_(alphas)

    clas_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, weights, reduction="sum"
    )
    return clas_loss


class RetinaNetLosses(nn.Module):
    def __init__(self, num_classes) -> None:
        super(RetinaNetLosses, self).__init__()
        self.n_c = num_classes

    def _encode_class(self, idxs):
        "one_hot encode targets such that 0 is the `background`"
        target = idxs.new_zeros(len(idxs), self.n_c).float()
        mask = idxs != 0
        i1s = torch.LongTensor(list(range(len(idxs))))
        target[i1s[mask], idxs[mask] - 1] = 1
        return target

    def classification_loss(self, targets, head_outputs, matches):
        # Calculate Classification Loss for `ClassSubnet` of  `RetinaNet`
        loss = []
        cls_logits = head_outputs["cls_preds"]
        targets = targets["labels"]

        for clas_tgt, clas_pred, m_idx in zip(targets, cls_logits, matches):

            fgs = m_idx >= 0

            m_idx.add_(1)
            clas_tgt = clas_tgt + 1
            clas_mask = m_idx >= 0
            clas_pred = clas_pred[clas_mask]

            clas_tgt = torch.cat([clas_tgt.new_zeros(1).long(), clas_tgt])
            clas_tgt = clas_tgt[matches[clas_mask]]
            clas_tgt = self._encode_class(clas_tgt, clas_pred.size(1))

            clas_loss = focal_loss(clas_pred, clas_tgt) / torch.clamp(
                fgs.sum(), min=1.0
            )
            loss.append(clas_loss)
        loss = sum(loss)
        return loss / len(targets)

    def regression_loss(self, targets, head_outputs, anchors, matches):
        # Calculate Regression Loss
        bbox_pred = head_outputs["bbox_preds"]
        loss = []

        for (tgt, bb_pred, ancs, m_idx,) in zip(targets, bbox_pred, anchors, matches):
            # no matches means there were no annotations in this image
            if m_idx.numel() == 0:
                continue

            # get the targets corresponding GT for each proposal
            matched = tgt["boxes"][m_idx.clamp(min=0)]

            # determine only the foreground indices, ignore the rest
            bbox_mask = m_idx >= 0
            if bbox_mask.sum() != 0:
                # select only the foreground boxes
                matched = matched[bbox_mask, :]
                bb_pred = bb_pred[bbox_mask, :]
                ancs = ancs[bbox_mask, :]
                # compute the regression targets
                bbox_targ = bbox_2_activ(matched, ancs)
                # compute the loss
                bb_loss = F.smooth_l1_loss(bb_pred, bbox_targ)
            else:
                bb_loss = 0.0

            loss.append(bb_loss)

        loss = sum(loss)
        return loss / max(1, len(targets))
