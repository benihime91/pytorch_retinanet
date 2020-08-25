from typing import *

import torch
import torch.nn as nn
from torch.functional import Tensor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.boxes import batched_nms, clip_boxes_to_image, remove_small_boxes

from .config import *
from .backbone import get_backbone
from .anchors import AnchorGenerator
from .layers import FPN, RetinaNetHead
from .utils import activ_2_bbox, ifnone

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

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        - num_classes   (int): number of output classes of the model (including the background).
                               0 is the background class.
        - backbone_kind (str): the network used to compute the features for the model. 
                               currently support only `Resnet` networks.
        - prior       (float): Prior prob for rare case (i.e. foreground) at the beginning of training.
        - pretrianed   (bool): Wether the backbone should be `pretrained` or not.
        - nms_thres   (float): Overlap threshold used for non-maximum suppression 
                               (suppress boxes with IoU >= this threshold)
        - score_thres (float): Minimum score threshold (assuming scores in a [0, 1] range.
        - max_detections_per_images(int): Number of proposals to keep after applying NMS.
        - freeze_bn   (bool): Wether to freeze the `BatchNorm` layers of the `BackBone` network.
        - anchor_generator(AnchorGenertor): Must be an instance of `AnchorGenerator`. 
                                            If None the default AnchorGenerator is used.
                                            see `config.py`
        - min_size (int)    : `minimum size` of the image to be rescaled before 
                               feeding it to the backbone.
        - max_size (int)    : `maximum size` of the image to be rescaled before 
                               feeding it to the backbone.
        - image_mean (List[float]): mean values used for input normalization.
        - image_std (List[float]): std values used for input normalization.

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
        image_mean: List[float] = None,
        image_std: List[float] = None,
        anchor_generator: Optional[AnchorGenerator] = None,
    ) -> None:

        super(Retinanet, self).__init__()

        num_classes = ifnone(num_classes, NUM_CLASSES)
        backbone_kind = ifnone(backbone_kind, BACKBONE)
        prior = ifnone(prior, PRIOR)
        pretrained = ifnone(pretrained, PRETRAINED_BACKBONE)
        nms_thres = ifnone(nms_thres, NMS_THRES)
        score_thres = ifnone(score_thres, SCORE_THRES)
        max_detections_per_images = ifnone(
            max_detections_per_images, MAX_DETECTIONS_PER_IMAGE
        )
        freeze_bn = ifnone(freeze_bn, FREEZE_BN)
        min_size = ifnone(min_size, MIN_IMAGE_SIZE)
        max_size = ifnone(max_size, MAX_IMAGE_SIZE)
        image_mean = ifnone(image_mean, MEAN)
        image_std = ifnone(image_std, STD)
        anchor_generator = ifnone(anchor_generator, AnchorGenerator())

        # The reason for the 0.05 is because that is what appears to be used by other systems as well,
        # such as faster rcnn and Detectron.
        assert (
            backbone_kind in __small__ + __big__
        ), f" Expected `backbone_kind` to be one of {__small__+__big__} got {backbone_kind}"

        # ------------------------------------------------------
        # Assemble `RetinaNet`
        # ------------------------------------------------------
        # # Instantiate `GeneralizedRCNNTransform` to resize inputs
        self.transform_inputs = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std
        )
        # Get the back bone of the Model
        self.backbone_kind = backbone_kind
        self.backbone = get_backbone(
            self.backbone_kind, pretrained, freeze_bn=freeze_bn
        )
        # # Grab the backbone output channels
        self.fpn_szs = self._get_backbone_ouputs()
        # # Instantiate the `FPN`
        self.fpn = FPN(
            self.fpn_szs[0], self.fpn_szs[1], self.fpn_szs[2], out_channels=256
        )
        # # Instantiate anchor Generator
        self.anchor_generator = anchor_generator
        self.num_anchors = self.anchor_generator.num_cell_anchors[0]
        # # Instantiate `RetinaNetHead`
        self.retinanet_head = RetinaNetHead(
            256, 256, self.num_anchors, num_classes, prior
        )

        # ------------------------------------------------------
        # Model Parameters
        # ------------------------------------------------------
        self.score_thres = score_thres
        self.nms_thres = nms_thres
        self.detections_per_img = max_detections_per_images
        self.num_classes = num_classes

    def _get_backbone_ouputs(self) -> List:
        if self.backbone_kind in __small__:
            fpn_szs = [
                self.backbone.backbone.layer2[1].conv2.out_channels,
                self.backbone.backbone.layer3[1].conv2.out_channels,
                self.backbone.backbone.layer4[1].conv2.out_channels,
            ]
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
        bg_clas: int = 0,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:

        "Process `outputs` and return the predicted bboxes, score, clas_labels above `detect_thresh`."

        class_logits = outputs.pop("cls_preds")
        bboxes = outputs.pop("bbox_preds")

        device = class_logits.device
        num_classes = class_logits.shape[-1]
        scores = torch.sigmoid(class_logits)

        # create labels for each score: here labels will be from 0 to num_classes
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        detections = torch.jit.annotate(List[Dict[str, Tensor]], [])

        all_boxes, all_scores, all_labels = [], [], []
        detections = []

        for bb_per_im, sc_per_im, lbl_per_im, ancs_per_im, im_sz in zip(
            bboxes, scores, labels, anchors, im_szs
        ):
            # Remove all predicitons corresponding to the background class from the predictions
            i = torch.min(torch.nonzero(lbl_per_im - bg_clas))
            bb_per_im, sc_per_im, lbl_per_im = (
                bb_per_im[i:],
                sc_per_im[i:],
                lbl_per_im[i:],
            )

            # Convret the activations of the Model into boiunding Bxes
            bb_per_im = activ_2_bbox(bb_per_im, ancs_per_im)
            # Clip boxes to Image
            bb_per_im = clip_boxes_to_image(bb_per_im, im_sz)
            # Remove Small Boxes
            keep = remove_small_boxes(bb_per_im, min_size=1e-02)
            bb_per_im, sc_per_im, lbl_per_im = (
                bb_per_im[keep],
                sc_per_im[keep],
                lbl_per_im[keep],
            )

            # batch everything, by making every class prediction be a separate instance
            bb_per_im = bb_per_im.reshape(-1, 4)
            sc_per_im = sc_per_im.reshape(-1)
            lbl_per_im = lbl_per_im.reshape(-1)

            # remove low scoring boxes
            lwl_thres = torch.nonzero(scores > self.score_thres).squeeze(1)
            bb_per_im, sc_per_im, lbl_per_im = (
                bb_per_im[lwl_thres],
                sc_per_im[lwl_thres],
                lbl_per_im[lwl_thres],
            )

            # non-max supression
            keep = batched_nms(bb_per_im, sc_per_im, lbl_per_im, self.nms_thres)
            # keep only the topk scoring preds
            keep = keep[: self.detections_per_img]
            bb_per_im, sc_per_im, lbl_per_im = (
                bb_per_im[keep],
                sc_per_im[keep],
                lbl_per_im[keep],
            )

            all_boxes.append(bb_per_im)
            all_scores.append(sc_per_im)
            all_labels.append(lbl_per_im)

        return all_boxes, all_scores, all_labels

    def _get_outputs(
        self, losses, detections
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:

        "if `training` return losses else return `detections`"
        if self.training:
            return losses
        else:
            return detections

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:

        if self.training and targets is None:
            raise ValueError("In training Model, `targets` must be given")

        # Grab the original Image sizes
        orig_im_szs = []

        # store the original img szs
        for im in images:
            val = im.shape[-2:]
            assert len(val) == 2
            orig_im_szs.append((val[0], val[1]))

        # Forward pass
        images, targets = self.transform_inputs(images, targets)
        feature_maps = self.backbone(images.tensors)
        feature_maps = self.fpn(feature_maps)
        anchors = self.anchor_generator(images, feature_maps)
        outputs = self.retinanet_head(feature_maps)

        # Instatiate loss and detections dictionary
        losses = {}
        detections = torch.jit.annotate(List[Dict[str, Tensor]], [])

        # Compute Loss or Detections
        if self.training:
            # Comput Loss
            losses = self.compute_loss(targets, outputs, anchors)
        else:
            # compute the detections
            with torch.no_grad():
                boxes, scores, labels = self.process_detections(
                    outputs, anchors, images.image_sizes
                )
                num_images = len(boxes)

                for i in range(num_images):
                    detections.append(
                        {"boxes": boxes[i], "labels": labels[i], "scores": scores[i],}
                    )

                detections = self.transform_inputs.postprocess(
                    detections, images.image_sizes, orig_im_szs
                )
        # Return Outputs
        return self._get_outputs(losses, detections)
