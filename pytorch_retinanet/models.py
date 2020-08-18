# From : https: // github.com/pytorch/vision/blob/master/torchvision/models/detection/roi_heads.py

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from torchvision.models import detection
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.boxes import clip_boxes_to_image, nms, remove_small_boxes

from .anchors import AnchorGenerator
from .layers import FPN, RetinaNetHead, get_backbone
from .utils import activ_2_bbox, matcher

__small__ = ['resnet18', 'resnet34']
__big__ = ['resnet50', 'resnet101', 'resnet101', 'resnet152']


class Retinanet(nn.Module):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    def __init__(self,
                 num_classes: int,
                 backbone_kind: str = 'resnet18',
                 prior: float = 0.01,
                 pretrained: bool = True,
                 nms_thres: float = 0.5,
                 score_thres: float = 0.5,
                 max_detections_per_images: int = 100) -> None:

        super(Retinanet, self).__init__()
        assert backbone_kind in __small__ + \
            __big__, f" Expected `backbone_kind` to be one of {__small__+__big__} got {backbone_kind}"

        # Get the back bone of the Model
        self.backbone = get_backbone(backbone_kind, pretrained)

        # Grab the backbone output channels
        if backbone_kind in __small__:
            self.fpn_szs = [
                self.backbone.backbone.layer2[1].conv2.out_channels,
                self.backbone.backbone.layer3[1].conv2.out_channels,
                self.backbone.backbone.layer4[1].conv2.out_channels,
            ]
        elif backbone_kind in __big__:
            self.fpn_szs = [
                self.backbone.backbone.layer2[2].conv3.out_channels,
                self.backbone.backbone.layer3[2].conv3.out_channels,
                self.backbone.backbone.layer4[2].conv3.out_channels,
            ]

        # Instantiate anchor Generator
        self.anchor_generator = AnchorGenerator()
        num_anchors = self.anchor_generator.num_cell_anchors[0]

        # Instantiate the `FPN`
        self.fpn = FPN(self.fpn_szs[0], self.fpn_szs[1],
                       self.fpn_szs[2], out_channels=256)

        # Instantiate `RetinaNetHead`
        self.retinanet_head = RetinaNetHead(
            256, num_classes, 256, num_anchors, prior)

        # Transoforms Input Images
        # Imagenet Mean & std
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.pre_tfms = GeneralizedRCNNTransform(600, 1200, mean, std)

        # Parameters
        self.score_thres = score_thres
        self.nms_thres = nms_thres
        self.detections_per_images = max_detections_per_images

    def return_outs(self, losses, detections):
        if self.training:
            return losses
        else:
            return detections

    def compute_retinanet_loss(self, targets, outputs, anchors):
        loss = self.retinanet_head.retinanet_focal_loss(
            targets, outputs, anchors)
        return loss

    def process_detections(self,
                           outputs: Dict[str, Tensor],
                           anchors: List[Tensor],
                           images_shapes: List[Tuple[int, int]]) -> Dict[str, Tensor]:

        # From : https: // github.com/pytorch/vision/blob/master/torchvision/models/detection/roi_heads.py
        # TODO : Make changes to make compatible with torchvision coco_evaluate script

        cls_preds = outputs.pop('logits')
        box_preds = outputs.pop('bboxes')

        device = cls_preds.device
        num_classes = cls_preds.shape[-1]

        # create labels for each score
        labels = torch.arange(num_classes, device=device)
        # make labels same shape as cls_preds: [batch_size, num_boxes , num_classes]
        labels = labels.view(1, -1).expand_as(cls_preds)

        detections = []

        for box_pred, cls_pred, lbl, anc, sz in \
                zip(box_preds, cls_preds, labels, anchors, images_shapes):

            # Transfrom the `retinanet_activations` into `bboxes`
            box_pred = activ_2_bbox(box_pred, anc)
            # Clip the `bboxes`
            box_pred = clip_boxes_to_image(box_pred, sz)

            all_boxes = []
            all_scores = []
            all_labels = []

            # Iterate over all the classes
            for cls_idx in range(num_classes):
                # Filter empty boxes
                keep_mask = remove_small_boxes(box_pred, min_size=1e-03)
                box_pred = box_pred[keep_mask]
                cls_pred = cls_pred[keep_mask]
                lbl = lbl[keep_mask]

                # Filter Detections using Score_threshold
                score_mask = torch.gt(cls_pred[:, cls_idx], self.score_thres)
                box_pred = box_pred[score_mask]
                cls_pred = cls_pred[score_mask, cls_idx]
                lbl = lbl[score_mask, cls_idx]

                # Do NMS
                keep_mask = nms(box_pred, cls_pred, self.nms_thres)

                # Keep top classes upto `detections_per_images`
                keep_mask = keep_mask[:self.detections_per_images]

                # Gather predictions
                box_pred, cls_pred, lbl = box_pred[keep_mask], cls_pred[keep_mask], lbl[keep_mask]

                all_boxes.append(box_pred)
                all_scores.append(cls_pred)
                all_labels.append(lbl)

            detections.append({
                'boxes': torch.cat(all_boxes,   dim=0),
                'scores': torch.cat(all_scores, dim=0),
                'labels': torch.cat(all_labels, dim=0),
            })

        return detections

    def forward(self, images, targets=None):
        if self.training:
            assert targets is not None, "If model is `training`, `targets` must be given."

        # Grab the original Image sizes
        original_image_sizes = []
        for img in images:
            sz = img.shape[-2:]
            assert len(sz) == 2, "Invalid `image_size`"
            original_image_sizes.append((sz[0], sz[1]))

        # Transform the Inputs
        images, targets = self.pre_tfms(images, targets)

        # Get the `feature_maps` from the `backbone`
        feature_maps: List[Tensor] = self.backbone(images.tensors)
        # Get `outputs` from `FPN`
        outputs = self.fpn(feature_maps)
        # Get `head_outputs` from `RetinaNetHead`
        head_outputs = self.retinanet_head(outputs)

        # print(head_outputs['bboxes'].shape)
        # create anchors
        anchors = self.anchor_generator(feature_maps)
        # print([anchors.shape for anchors in anchors])

        losses = {}
        detections = {}

        if self.training:
            # Compute Losses
            losses = self.compute_retinanet_loss(
                targets, head_outputs, anchors)
        else:
            # Compute Detections
            detections = self.process_detections(
                head_outputs, anchors, images.image_sizes)

            detections = self.pre_tfms.postprocess(
                detections, images.image_sizes, original_image_sizes)

        return self.return_outs(losses, detection)
