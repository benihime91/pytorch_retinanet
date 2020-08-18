from typing import *

import torch
import torch.nn as nn
from torch.functional import Tensor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.boxes import clip_boxes_to_image, nms, remove_small_boxes

from .anchors import AnchorGenerator
from .layers import FPN, RetinaNetHead, get_backbone
from .utils import activ_2_bbox

__small__ = ['resnet18', 'resnet34']
__big__ = ['resnet50', 'resnet101', 'resnet101', 'resnet152']

# TODO : Make changes to make compatible with `torchvision` coco_evaluate script


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
                 score_thres: float = 0.05,
                 max_detections_per_images: int = 300,
                 freeze_bn: bool = True) -> None:

        # The reason for the 0.05 is because that is what appears to be used by other systems as well,
        # such as faster rcnn and Detectron.

        super(Retinanet, self).__init__()
        assert backbone_kind in __small__ + \
            __big__, f" Expected `backbone_kind` to be one of {__small__+__big__} got {backbone_kind}"

        # Instantiate `GeneralizedRCNNTransform to Resize Images`
        # Transoforms Input Images
        # Imagenet Mean & std
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.pre_tfms = GeneralizedRCNNTransform(600, 1200, mean, std)

        # Get the back bone of the Model
        self.backbone = (
            get_backbone(
                backbone_kind,
                pretrained,
                freeze_bn=freeze_bn))

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

        # Instantiate the `FPN`
        self.fpn = FPN(self.fpn_szs[0], self.fpn_szs[1],
                       self.fpn_szs[2], out_channels=256)

        # Instantiate anchor Generator
        self.anchor_generator = AnchorGenerator()
        self.num_anchors = self.anchor_generator.num_cell_anchors[0]

        # Instantiate `RetinaNetHead`
        self.retinanet_head = RetinaNetHead(
            256, num_classes, 256, self.num_anchors, prior)

        # Parameters
        self.score_thres = score_thres
        self.nms_thres = nms_thres
        self.detections_per_images = max_detections_per_images

    def compute_retinanet_loss(self, targets, outputs, anchors):
        loss = self.retinanet_head.retinanet_focal_loss(
            targets, outputs, anchors)
        return loss

    def process_detections(self,
                           outputs: Dict[str, Tensor],
                           anchors: List[Tensor],
                           image_shapes: List[Tuple[int, int]]) -> Dict[str, Tensor]:

        # # Modified From :
        # https: // github.com/pytorch/vision/blob/master/torchvision/models/detection/roi_heads.py

        # shape: (None, H*W*num_anchors, num_classes)
        pred_classes = outputs.pop('logits')
        pred_boxes = outputs.pop('bboxes')  # shape: (None, H*W*num_anchors, 4)

        device = pred_classes.device
        num_classes = pred_classes.shape[-1]

        # create pred_labels for each score
        pred_labels = torch.arange(num_classes, device=device)
        # make pred_labels same shape as pred_classes: [batch_size, num_boxes , num_classes]
        pred_labels = pred_labels.view(1, -1).expand_as(pred_classes)

        detections = []

        for pred_box, pred_cls, pred_lbl, anc, sz in \
                zip(pred_boxes, pred_classes, pred_labels, anchors, image_shapes):

            # Transfrom the `retinanet_activations` into `bboxes`
            pred_box = activ_2_bbox(pred_box, anc)
            # Clip the `bboxes`
            pred_box = clip_boxes_to_image(pred_box, sz)

            all_boxes = []
            all_scores = []
            all_labels = []

            # Iterate over all the classes
            for cls_idx in range(num_classes):

                # Filter Detections using Score_threshold: Remove Low Scoring Boxes
                inds = torch.gt(pred_cls[:, cls_idx], self.score_thres)
                # Filter out the cls_idx
                pred_box_per_cls, pred_cls_per_cls, pred_lbl_per_cls = \
                    pred_box[inds], pred_cls[inds,
                                             cls_idx], pred_lbl[inds, cls_idx]

                ######################################################################
                ########## Compute Detections for Single Class : cls_idx #############
                ######################################################################
                # 1. Filter empty boxes
                keep_mask = remove_small_boxes(
                    pred_box_per_cls, min_size=1e-02)
                pred_box_per_cls, pred_cls_per_cls, pred_lbl_per_cls = \
                    pred_box_per_cls[keep_mask], pred_cls_per_cls[keep_mask], pred_lbl_per_cls[keep_mask]

                # 2. Do NMS
                keep_mask = nms(
                    pred_box_per_cls,
                    pred_cls_per_cls,
                    self.nms_thres)

                # 3. Keep top classes upto `detections_per_images`
                keep_mask = keep_mask[:self.detections_per_images]

                # 4. Gather predictions
                pred_box_per_cls, pred_cls_per_cls, pred_lbl_per_cls = \
                    pred_box_per_cls[keep_mask], pred_cls_per_cls[keep_mask], pred_lbl_per_cls[keep_mask]

                # Update `ouputs` list
                all_boxes.append(pred_box_per_cls)
                all_scores.append(pred_cls_per_cls)
                all_labels.append(pred_lbl_per_cls)

            # Update Detection Dictionary
            detections.append({
                'boxes':  torch.cat(all_boxes, dim=0),
                'scores': torch.cat(all_scores, dim=0),
                'labels': torch.cat(all_labels, dim=0),
            })

        return detections

    def forward(self, images: Tensor, targets: Dict[str, Tensor] = None):
        "returns `detections` if model is `eval` else returns `loss`"

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

        # create anchors
        anchors = self.anchor_generator(outputs)

        losses = {}
        detections = {}

        if self.training:
            # Compute Losses
            losses = self.compute_retinanet_loss(
                targets, head_outputs, anchors)

            return losses

        else:
            with torch.no_grad():
                # Compute Detections
                detections = self.process_detections(
                    head_outputs, anchors, images.image_sizes)

                detections = self.pre_tfms.postprocess(
                    detections, images.image_sizes, original_image_sizes)

            return detections
