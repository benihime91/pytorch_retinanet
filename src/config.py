from typing import *

# ----------------------------------------------------------------------------- #
# Config definition
# ----------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------- #
# INPUT
# ----------------------------------------------------------------------------- #
# `Mean values` used for input normalization.
MEAN: List[float] = [0.485, 0.456, 0.406]
# `STD values` used for input normalization.
STD: List[float] = [0.229, 0.224, 0.225]
# Size of the smallest side of the image during training
MIN_IMAGE_SIZE: int = 600
# Maximum size of the side of the image during training
MAX_IMAGE_SIZE: int = 600


# ----------------------------------------------------------------------------- #
# # Anchor generator options
# ----------------------------------------------------------------------------- #
# Anchor sizes (i.e. sqrt of area) in absolute pixels w.r.t. the network input.
ANCHOR_SIZES: List[float] = [
    [x, x * 2 ** (1 / 3), x * 2 ** (2 / 3)] for x in [32, 64, 128, 256, 512]
]

# A list of float value representing the strides for each feature
# map in the feature pyramid.
ANCHOR_STRIDES: List[int] = [8, 16, 32, 64, 128]

# Anchor aspect ratios. For each area given in `SIZES`, anchors with different aspect
# ratios are generated by an anchor generator.
ANCHOR_ASPECT_RATIOS: List[float] = [0.5, 1.0, 2.0]

# Relative offset between the center of the first anchor and the top-left corner of the image
# Value has to be in [0, 1). Recommend to use 0.5, which means half stride.
# The value is not expected to affect model accuracy.
ANCHOR_OFFSET: float = 0.0


# ----------------------------------------------------------------------------- #
# RetinaNet Head
# ----------------------------------------------------------------------------- #
NUM_CLASSES: int = 80
# This is the number of foreground classes.

# The network used to compute the features for the model.
# Should be one of ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet101', 'resnet152'].
BACKBONE: str = "resnet50"
# Wether the backbone should be pretrained or not,. If true loads `pre-trained` weights
PRETRAINED_BACKBONE: bool = True

# Prior prob for rare case (i.e. foreground) at the beginning of training.
# This is used to set the bias for the logits layer of the classifier subnet.
# This improves training stability in the case of heavy class imbalance.
PRIOR: float = 0.01

# Wether to freeze `BatchNormalization` layers of `backbone`
FREEZE_BN: bool = True

# Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
BBOX_REG_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

# Inference cls score threshold, only anchors with score > INFERENCE_TH are
# considered for inference (to improve speed)
SCORE_THRES: float = 0.05
NMS_THRES: float = 0.5
MAX_DETECTIONS_PER_IMAGE: int = 300

# IoU overlap ratio bg & fg for labeling anchors.
# Anchors with < bg are labeled negative (0)
# Anchors  with >= bg and < fg are ignored (-1)
# Anchors with >= fg are labeled positive (1)
IOU_THRESHOLDS_FOREGROUND: float = 0.5
IOU_THRESHOLDS_BACKGROUND: float = 0.4

# Loss parameters
FOCAL_LOSS_GAMMA: float = 2.0
FOCAL_LOSS_ALPHA: float = 0.25
SMOOTH_L1_LOSS_BETA: float = 0.1
