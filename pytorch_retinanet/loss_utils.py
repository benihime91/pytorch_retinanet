from typing import *
import torch
import torch.nn.functional as F
from torch import device, dtype, nn


def one_hot(labels: torch.Tensor, num_classes: int, device: torch.device = torch.device('cpu')):
    """
    Converts an Integer Label to to OneHot Tensors

    Args:
        labels (torch.Tensor): tensor with labels of shape: `(N, *)`
                               where each value is an integer corresponding to
                               the correct classes.
        num_classes (int)    : total number of classes
        device      (str)    : one of ['cpu', 'cuda:0']
    """
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:], device=device)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
