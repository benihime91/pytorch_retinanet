from typing import *
import torch
import torch.nn as nn
from torch.functional import Tensor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.boxes import clip_boxes_to_image, nms, remove_small_boxes
from .anchors import AnchorGenerator
from .config import *
from .layers import FPN, RetinaNetHead, get_backbone
from .utilities import ifnone
from .utils import activ_2_bbox

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
        - num_classes   (int): number of output classes of the model (excluding the background).
        - backbone_kind (str): the network used to compute the features for the model. Currently support only `Resnet` networks.
        - prior       (float): Prior prob for rare case (i.e. foreground) at the beginning of training.
        - pretrianed   (bool): Wether the backbone should be `pretrained` or not.
        - nms_thres   (float): Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)
        - score_thres (float): Minimum score threshold (assuming scores in a [0, 1] range.
        - max_detections_per_images(int): Number of proposals to keep after applying NMS.
        - freeze_bn   (bool): Wether to freeze the `BatchNorm` layers of the `BackBone` network.
        - anchor_generator(AnchorGenertor): Must be an instance of `AnchorGenerator`. If None the default AnchorGenerator is used.
                                            see `config.py`
        - min_size (int)    : `minimum size` of the image to be rescaled before feeding it to the backbone.
        - max_size (int)    : `maximum size` of the image to be rescaled before feeding it to the backbone.
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

        super(Retinanet, self).__init__()
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
        # Parameters
        # ------------------------------------------------------
        self.score_thres = score_thres
        self.nms_thres = nms_thres
        self.detections_per_images = max_detections_per_images

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

    @staticmethod
    def process_detections(
        outputs: Dict[str, Tensor], anchors: List[Tensor], image_shapes
    ) -> List[Dict[str, Tensor]]:

        cls_preds = outputs.pop("cls_preds")
        bbox_preds = outputs.pop("bbox_preds")

        device = cls_preds.device()
        num_classes = cls_preds.shape[-1]

