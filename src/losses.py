from typing import *
import torch
import torch.nn.functional as F
from torch.functional import Tensor
from .config import *
from .utils import bbox_2_activ


def smooth_l1_loss(inp: Tensor, targs: Tensor, reduction: str = "mean") -> Tensor:
    """
    Computes `Smooth_L1_Loss`
    """
    return F.smooth_l1_loss(inp, targs, reduction=reduction)


def focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = FOCAL_LOSS_ALPHA,
    gamma: float = FOCAL_LOSS_GAMMA,
    reduction: str = "mean",
) -> Tensor:
    """
    From: https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .

    Args:
        inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        reduction:  'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def classification_loss(
    targets: List[Dict[str, Tensor]],
    outputs: Dict[str, Tensor],
    matched_idxs: List[Tensor],
) -> Tensor:
    # ---------------------------------------------------------------
    # Calculate Classification Loss for `ClassSubnet` of  `RetinaNet`
    # ---------------------------------------------------------------
    cls_preds = outputs["cls_preds"]
    device = [c.device for c in cls_preds][0]

    classification_loss = torch.tensor(0.0, device=device)

    for tgt, cls_pred, m_idx in zip(targets, cls_preds, matched_idxs):
        # no matched_idxs means there were no annotations in this image
        if m_idx.numel() == 0:
            gt_targs = torch.zeros_like(cls_pred)
            valid_idxs = torch.arange(cls_pred.shape[0])
            num_foreground = torch.tensor(0.0, device=device)
        else:
            # determine only the foreground
            foreground_idxs_ = m_idx >= 0
            num_foreground = foreground_idxs_.sum()
            gt_targs = torch.zeros_like(cls_pred, device=device)

            # create the target classification
            gt_targs[
                foreground_idxs_, tgt["labels"][m_idx[foreground_idxs_]]
            ] = torch.tensor(1.0, device=device)
            # find indices for which anchors should be ignored
            valid_idxs = m_idx != IGNORE_IDX

        # compute the classification loss
        classification_loss += focal_loss(
            cls_pred[valid_idxs], gt_targs[valid_idxs], reduction="sum"
        ).cpu() / max(1, num_foreground.cpu())

    return classification_loss / len(targets)


def regression_loss(
    targets: List[Dict[str, Tensor]],
    outputs: Dict[str, Tensor],
    anchors: List[Tensor],
    matched_idxs: List[Tensor],
) -> Tensor:
    # ---------------------------------------------------------------
    # Calculate Regression Loss
    # ---------------------------------------------------------------
    bbox_preds = outputs["bbox_preds"]
    device = [bbox.device for bbox in bbox_preds][0]

    loss = torch.tensor(0.0, device=device)

    for tgt, bbox, anc, idxs in zip(targets, bbox_preds, anchors, matched_idxs):
        # no matched_idxs means there were no annotations in this image
        if idxs.numel() == 0:
            continue

        matched_gts = tgt["boxes"][idxs.clamp(min=0)]
        # determine only the foreground indices, ignore the rest
        foreground_idxs_ = idxs >= 0
        num_foreground = foreground_idxs_.sum()

        # select only the foreground boxes
        matched_gts = torch.tensor(matched_gts[foreground_idxs_, :], device=device)
        bbox = torch.tensor(bbox[foreground_idxs_, :], device=device)
        anc = torch.tensor(anc[foreground_idxs_, :], device=device)

        # compute the regression targets
        targs = bbox_2_activ(matched_gts, anc)

        # Compute loss
        loss += smooth_l1_loss(bbox, targs, reduction="sum") / max(
            1, num_foreground.cpu()
        )

    return loss / max(1, len(targets))
