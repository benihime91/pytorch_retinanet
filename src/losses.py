from typing import *
import torch.nn.functional as F
from torch.functional import Tensor
from .config import *


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
