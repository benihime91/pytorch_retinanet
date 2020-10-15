import logging
from typing import *

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import boxes as ops

from .anchors import AnchorGenerator
from .backbone import get_backbone
from .box_utils import activ_2_bbox
from .config import *
from .layers import FeaturePyramid, RetinaNetHead
from .utilities import ifnone

__small__ = ["resnet18", "resnet34"]
__big__ = ["resnet50", "resnet101", "resnet101", "resnet152"]


class Retinanet(nn.Module):
    """
    Implement RetinaNet in :paper:`RetinaNet`.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the `classification` and `regression` losses for
    the `RetinaNet` `classSubnet` & `BoxSubnet` repectively.

    For infererence, use `.predict` 
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        - num_classes   (int): number of output classes of the model (excluding the background).

        - backbone_kind (str): the network used to compute the features for the model.
                               currently support only `Resnet` networks.
        - prior       (float): Prior prob for rare case (i.e. foreground) at the beginning of training.
        - pretrianed   (bool): Wether the backbone should be `pretrained` or not.
        - nms_thres   (float): Overlap threshold used for non-maximum suppression
                               (suppress boxes with IoU >= this threshold).
        - score_thres (float): Minimum score threshold (assuming scores in a [0, 1] range.
        - max_detections_per_images(int): Number of proposals to keep after applying NMS.
        - freeze_bn   (bool) : Wether to freeze the `BatchNorm` layers of the `BackBone` network.
        - anchor_generator(AnchorGenertor): Must be an instance of `AnchorGenerator`.
                                            If None the default AnchorGenerator is used.
                                            see `config.py`
        - min_size (int)     : `minimum size` of the image to be rescaled before
                               feeding it to the backbone.
        - max_size (int)     : `maximum size` of the image to be rescaled before
                               feeding it to the backbone.
        - image_mean (List[float]): mean values used for input normalization.
        - image_std (List[float]) : std values used for input normalization.

    >>> For default values see `config.py`
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        backbone_kind: Optional[str] = None,
        prior: Optional[float] = None,
        pretrained: Optional[bool] = None,
        nms_thres: Optional[float] = None,
        score_thres: Optional[float] = None,
        max_detections_per_images: Optional[int] = None,
        freeze_bn: Optional[bool] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        anchor_generator: Optional[AnchorGenerator] = None,
        logger=None,
    ) -> None:

        super(Retinanet, self).__init__()

        # Set Parameters
        num_classes = ifnone(num_classes, NUM_CLASSES)
        backbone_kind = ifnone(backbone_kind, BACKBONE)
        prior = ifnone(prior, PRIOR)
        pretrained = ifnone(pretrained, PRETRAINED_BACKBONE)
        nms_thres = ifnone(nms_thres, NMS_THRES)
        score_thres = ifnone(score_thres, SCORE_THRES)
        max_detections_per_images = ifnone(max_detections_per_images, MAX_DETECTIONS_PER_IMAGE)
        freeze_bn = ifnone(freeze_bn, FREEZE_BN)
        min_size = ifnone(min_size, MIN_IMAGE_SIZE)
        max_size = ifnone(max_size, MAX_IMAGE_SIZE)
        image_mean = ifnone(image_mean, MEAN)
        image_std = ifnone(image_std, STD)
        anchor_generator = ifnone(anchor_generator, AnchorGenerator())
        logger = ifnone(logger, logging.getLogger(__name__))
        logger.name = __name__

        if backbone_kind not in __small__ + __big__:
            _prompt = f"Expected `backbone_kind` to be one of {__small__+__big__} got {backbone_kind}"
            raise ValueError(_prompt)

        # Instantiate modules for RetinaNet
        self.backbone_kind = backbone_kind
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        self.backbone = get_backbone(backbone_kind, pretrained, freeze_bn=freeze_bn)
        fpn_szs = self._get_backbone_ouputs()
        self.fpn = FeaturePyramid(fpn_szs[0], fpn_szs[1], fpn_szs[2], 256)
        self.anchor_generator = anchor_generator
        num_anchors = self.anchor_generator.num_cell_anchors[0]
        self.retinanet_head = RetinaNetHead(256, 256, num_anchors, num_classes, prior)

        # Parameters for detection
        self.score_thres        = score_thres
        self.nms_thres          = nms_thres
        self.detections_per_img = max_detections_per_images
        self.num_classes        = num_classes

        # Log some information
        logger.info(f"BACKBONE     : {backbone_kind}")
        logger.info(f"INPUT_PARAMS : MAX_SIZE={max_size}, MIN_SIZE={min_size}")
        logger.info(f"NUM_CLASSES  : {self.num_classes}")

    def _get_backbone_ouputs(self) -> List:
        if self.backbone_kind in __small__:
            fpn_szs = [
                self.backbone.backbone.layer2[1].conv2.out_channels,
                self.backbone.backbone.layer3[1].conv2.out_channels,
                self.backbone.backbone.layer4[1].conv2.out_channels,
            ]
            return fpn_szs

        elif self.backbone_kind in __big__:
            fpn_szs = [
                self.backbone.backbone.layer2[2].conv3.out_channels,
                self.backbone.backbone.layer3[2].conv3.out_channels,
                self.backbone.backbone.layer4[2].conv3.out_channels,
            ]
            return fpn_szs

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        outputs: Dict[str, Tensor],
        anchors: List[Tensor],
    ) -> Dict[str, Tensor]:
        return self.retinanet_head.compute_loss(targets, outputs, anchors)

    def process_detections(
        self,
        outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        im_szs: List[Tuple[int, int]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        " Process `outputs` and return the predicted bboxes, score, clas_labels above `detect_thres` "

        class_logits = outputs.pop("cls_preds")
        bboxes = outputs.pop("bbox_preds")
        scores = torch.sigmoid(class_logits)

        device = class_logits.device
        num_classes = class_logits.shape[-1]

        # create labels for each score
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        detections = torch.jit.annotate(List[Dict[str, Tensor]], [])

        for bb_per_im, sc_per_im, ancs_per_im, im_sz, lbl_per_im in zip(bboxes, scores, anchors, im_szs, labels):
            
            all_boxes = []
            all_scores = []
            all_labels = []
            # convert the activation i.e, outputs of the model to bounding boxes
            bb_per_im = activ_2_bbox(bb_per_im, ancs_per_im)
            # clip the bounding boxes to the image size
            bb_per_im = ops.clip_boxes_to_image(bb_per_im, im_sz)

            # Iterate over each `cls_idx` in `num_classes` and do nms
            # to each class individually
            for cls_idx in range(num_classes):
                # remove low predicitons with scores < score_thres
                #  and grab the predictions corresponding to the cls_idx
                inds = torch.gt(sc_per_im[:, cls_idx], self.score_thres)
                bb_per_cls, sc_per_cls, lbl_per_cls = (
                    bb_per_im[inds],
                    sc_per_im[inds, cls_idx],
                    lbl_per_im[inds, cls_idx],
                )
                # remove boxes that are too small ~(1-02)
                keep = ops.remove_small_boxes(bb_per_cls, min_size=1e-2)
                bb_per_cls, sc_per_cls, lbl_per_cls = (
                    bb_per_cls[keep],
                    sc_per_cls[keep],
                    lbl_per_cls[keep],
                )
                # compute non max supression to supress overlapping boxes
                keep = ops.nms(bb_per_cls, sc_per_cls, self.nms_thres)
                bb_per_cls, sc_per_cls, lbl_per_cls = (
                    bb_per_cls[keep],
                    sc_per_cls[keep],
                    lbl_per_cls[keep],
                )

                all_boxes.append(bb_per_cls)
                all_scores.append(sc_per_cls)
                all_labels.append(lbl_per_cls)

            # Convert to tensors
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # model is going to predict classes which are going to be in the range of [0, num_classes]
            # 0 is reserved for the background class for which no loss is calculate , so
            # we will add 1 to all the class_predictions to shift the predicitons range from
            # [0, num_classes) -> [1, num_classes]
            all_labels = all_labels + 1

            # Sort by scores and
            # Grab the idxs from the corresponding to the topk predictions
            _, topk_idxs = all_scores.sort(descending=True)
            topk_idxs = topk_idxs[: self.detections_per_img]
            all_boxes, all_scores, all_labels = (
                all_boxes[topk_idxs],
                all_scores[topk_idxs],
                all_labels[topk_idxs],
            )

            detections.append({"boxes": all_boxes, "scores": all_scores, "labels": all_labels,})
        return detections

    def predict(self, images: List[Tensor]) -> List[Dict[str, Tensor]]:
        """
        Computes predictions for the given model
        """
        #set model to eval
        if self.training :
            self.training = False
        
        targets = None
        # get the original image sizes
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        
        # Foward pass of the Model
        images, targets = self.transform(images, targets)
        feature_maps    = self.backbone(images.tensors)
        feature_maps    = self.fpn(feature_maps)
        outputs         = self.retinanet_head(feature_maps)
        anchors         = self.anchor_generator(images, feature_maps)
        
        detections       = torch.jit.annotate(List[Dict[str, Tensor]], [])
        #compute the detections
        detections       = self.process_detections(outputs, anchors, images.image_sizes)
        final_detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return final_detections

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]]) -> Dict[str, Tensor]:
        """
        Computes the loss of the model
        """
        # Foward pass of the Model
        images, targets = self.transform(images, targets)
        feature_maps    = self.backbone(images.tensors)
        feature_maps    = self.fpn(feature_maps)
        outputs         = self.retinanet_head(feature_maps)
        # Generate anchors for the images
        anchors         = self.anchor_generator(images, feature_maps)
        # store losses
        losses = {}
        losses = self.compute_loss(targets, outputs, anchors)
        return losses
