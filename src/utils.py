import importlib
from typing import *

import torch
from torch.functional import Tensor
from torchvision.ops.boxes import box_iou

from .config import *


def collate_fn(batch):
    "`collate_fn` for pytorch obj_detection dataloader"
    return tuple(zip(*batch))


def ifnone(a: Any, b: Any) -> Any:
    """`a` if `a` is not None, otherwise `b`"""
    if a is not None:
        return a
    else:
        return b


# https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def convert_tlbr_2_cthw(boxes: Tensor) -> Tensor:
    "Convert top/left bottom/right format `boxes` to center/size corners."
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = boxes[:, 2:] - boxes[:, :2]
    return torch.cat([center, sizes], 1)


def convert_cthw_2_tlbr(boxes: Tensor) -> Tensor:
    "Convert center/size format `boxes` to top/left bottom/right corners."
    top_left = boxes[:, :2] - boxes[:, 2:] / 2
    bot_right = boxes[:, :2] + boxes[:, 2:] / 2

    return torch.cat([top_left, bot_right], 1)


def bbox_2_activ(bboxes: Tensor, anchors: Tensor) -> Tensor:
    "Return the target of the model on `anchors` for the `bboxes`."
    # Anchors & bboxes are of the forms : XYXY
    # Convert to XYWH
    bboxes, anchors = convert_tlbr_2_cthw(bboxes), convert_tlbr_2_cthw(anchors)

    t_centers = (bboxes[..., :2] - anchors[..., :2]) / anchors[..., 2:]

    t_sizes = torch.log(bboxes[..., 2:] / anchors[..., 2:] + 1e-8)

    deltas = torch.cat([t_centers, t_sizes], -1).div_(
        bboxes.new_tensor([BBOX_REG_WEIGHTS])
    )
    return deltas


def activ_2_bbox(activations: Tensor, anchors: Tensor) -> Tensor:
    "Converts the `activations` of the `model` to bounding boxes."
    # Anchors are of the form: XYXY & activations are of the form XYWH
    # Convert anchors to XYWH
    anchors = convert_tlbr_2_cthw(anchors)

    activations.mul_(activations.new_tensor([BBOX_REG_WEIGHTS]))

    centers = anchors[..., 2:] * activations[..., :2] + anchors[..., :2]
    sizes = anchors[..., 2:] * torch.exp(activations[..., :2])

    return convert_cthw_2_tlbr(torch.cat([centers, sizes], -1))


def matcher(
    anchors: Tensor, targets: Tensor, match_thr: float = None, back_thr: float = None
):
    """
    Match `anchors` to targets. -1 is match to background, -2 is ignore.
    """
    # From: https: // github.com/fastai/course-v3/blob/9c83dfbf9b9415456c9801d299d86e099b36c86d/nbs/dl2/pascal.ipynb
    # - for each anchor we take the maximum overlap possible with any of the targets.
    # - if that maximum overlap is less than 0.4, we match the anchor box to background,
    # the classifier's target will be that class.
    # - if the maximum overlap is greater than 0.5, we match the anchor box to that ground truth object.
    # The classifier's target will be the category of that target.
    # - if the maximum overlap is between 0.4 and 0.5, we ignore that anchor in our loss computation.
    match_thr = ifnone(match_thr, IOU_THRESHOLDS_FOREGROUND)
    back_thr = ifnone(back_thr, IOU_THRESHOLDS_BACKGROUND)
    assert match_thr > back_thr

    matches = anchors.new(anchors.size(0)).zero_().long() - 2

    if targets.numel() == 0:
        return matches

    # Calculate IOU between given targets & anchors
    iou_vals = box_iou(targets, anchors)
    # Grab the best ground_truth overlap
    vals, idxs = iou_vals.max(dim=0)
    # Grab the idxs
    matches[vals < back_thr] = -1
    matches[vals > match_thr] = idxs[vals > match_thr]
    return matches
