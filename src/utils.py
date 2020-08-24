import importlib
import math
from typing import *
import torch
from torch import flatten
from torch.functional import Tensor

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
    obj_path      = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name      = obj_path_list[0]
    module_obj    = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def bbox_2_activ(bboxes: Tensor, anchors: Tensor) -> Tensor:
    """
    Convert `ground_truths` to match the model `activations` to calculate `loss`.
    """
    if anchors.device != bboxes.device:
        anchors.to(bboxes.device)
    
    # Unpack elements
    anchor_x1 = anchors[:,0].unsqueeze(1)
    anchor_x2 = anchors[:,1].unsqueeze(1)
    anchor_y1 = anchors[:,2].unsqueeze(1)
    anchor_y2 = anchors[:,3].unsqueeze(1)

    bbox_x1 = bboxes[:,0].unsqueeze(1)
    bbox_x2 = bboxes[:,1].unsqueeze(1)
    bbox_y1 = bboxes[:,2].unsqueeze(1)
    bbox_y2 = bboxes[:,3].unsqueeze(1)  

    # Convert from tlbr to cthw
    a_width    = anchor_x2 - anchor_x1
    a_height   = anchor_y1 - anchor_y1
    a_center_x = anchor_x1 + 0.5 * a_width
    a_center_y = anchor_y1 + 0.5 * a_height

    b_width    = bbox_x2 - bbox_x1
    b_height   = bbox_y1 - bbox_y1
    b_center_x = bbox_x1 + 0.5 * b_width
    b_center_y = bbox_y1 + 0.5 * b_height
    
    # Compute Offsets
    t_x = (b_center_x - a_center_x)/a_width
    t_y = (b_center_y - a_center_y)/a_height
    t_w = torch.log(b_width / a_width + 1e-08)
    t_h = torch.log(b_height / a_height + 1e-08)
    return torch.cat((t_x, t_y, t_w, t_h), 1).div_(bboxes.new_tensor([BBOX_REG_WEIGHTS]))

def activ_2_bbox(activations: Tensor, anchors: Tensor):
    "Converts the `activations` of the `model` to bounding boxes."
    # Gather in the same device
    if anchors.device != activations.device:
        anchors = anchors.to(activations.device)
    # Convert anchors from tlbr to cthw
    widths  = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x   = anchors[:, 0] + 0.5 * widths
    ctr_y   = anchors[:, 1] + 0.5 * heights

    activations.mul_(activations.new_tensor([BBOX_REG_WEIGHTS]))  # multiply activation with weights
    dx = activations[:, 0::4]
    dy = activations[:, 1::4]
    dw = activations[:, 2::4]
    dh = activations[:, 3::4]

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=math.log(1000./16))
    dh = torch.clamp(dh, max=math.log(1000./16))

    # Calculate predictied bbox co-ordinates : format cthw
    center_x = dx * widths[:, None] + ctr_x[:, None]
    center_y = dy * heights[:, None] + ctr_y[:, None]
    width = torch.exp(dw) * widths[:, None]
    height = torch.exp(dh) * heights[:, None]
    
    # Convert from cthw to tlbr
    bbox_1 = center_x - torch.tensor(0.5, dtype=center_x.dtype, device=width.device) * width
    bbox_2 = center_y - torch.tensor(0.5, dtype=center_y.dtype, device=height.device) * height
    bbox_3 = center_x + torch.tensor(0.5, dtype=center_x.dtype, device=width.device) * width
    bbox_4 = center_y + torch.tensor(0.5, dtype=center_y.dtype, device=height.device) * height   
    # Create the Predicitons
    bbox = torch.stack((bbox_1, bbox_2, bbox_3, bbox_4), dim=2).flatten(1)    
    return bbox


def matcher(anchors: Tensor, targets: Tensor, match_thr: float = None, back_thr: float = None):
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
    back_thr  = ifnone(back_thr, IOU_THRESHOLDS_BACKGROUND)
    assert (match_thr > back_thr)

    matches = anchors.new(anchors.size(0)).zero_().long() - 2
    if targets.numel() == 0:
        return matches

    # Calculate IOU between given targets & anchors
    iou_vals   = compute_IOU(anchors, targets)
    # Grab the best ground_truth overlap
    vals, idxs = iou_vals.max(dim=1)
    # Grab the idxs
    matches[vals < back_thr] = -1
    matches[vals > match_thr] = idxs[vals > match_thr]
    return matches


def compute_IOU(anchors, targets):
    "Compute the IoU values of `anchors` by `targets`."
    inter          = intersection(anchors, targets)
    anc_sz, tgt_sz = anchors[:, 2] * anchors[:, 3], targets[:, 2] * targets[:, 3]
    union          = anc_sz.unsqueeze(1) + tgt_sz.unsqueeze(0) - inter
    return inter / (union + 1e-8)


def intersection(anchors, targets):
    "Compute the sizes of the intersections of `anchors` by `targets`."
    a, t       = anchors.size(0), targets.size(0)
    ancs, tgts = (anchors.unsqueeze(1).expand(a, t, 4), targets.unsqueeze(0).expand(a, t, 4))
    top_left_i  = torch.max(ancs[..., :2], tgts[..., :2])
    bot_right_i = torch.min(ancs[..., 2:], tgts[..., 2:])
    sizes       = torch.clamp(bot_right_i - top_left_i, min=0)
    return sizes[..., 0] * sizes[..., 1]

