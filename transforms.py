# -----------------------------------------------------------------------------
# Albumentation Transforms
# -----------------------------------------------------------------------------
# Specify Data Transformations in this files:
# Currently supports `Albumentations` transforms.
# Check : https://github.com/albumentations-team/albumentations
# for list of all available transforms and `API` docs.

# NB:
# 1. Images should be transformed to have `pixel` values between 0, 1
# 2. Images can also be of different shapes.
# 3. Images should have 3 `channels`
# 4. Each Image should contain a `Bounding Box` & `Correponding Class`
import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from typing import *

# -----------------------------------------------------------------------------
# `Default` Parameters
# -----------------------------------------------------------------------------
## Do not Change These Params
BBOX_PARAMS = "pascal_voc"
LABEL_FIELDS = "class_labels"

DEFAULT_TRANSFORMS: list = [
    A.ToFloat(max_value=255.0, always_apply=True),
    ToTensorV2(always_apply=True),
]


# -----------------------------------------------------------------------------
# `Train` Transformations
# -----------------------------------------------------------------------------
TRAIN_TRANSFORMATIONS: list = [
    # Define the transformations for Training set here:
    # If No Transformations to apply leave empty list
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.35),
    A.CLAHE(),
    A.IAASharpen(),
    A.IAAPerspective(),
    A.OneOf([A.ShiftScaleRotate(), A.Rotate(limit=60),], p=1.0),
    A.OneOf([A.RandomShadow(), A.RandomBrightnessContrast(), A.Cutout()], p=0.5),
]
# -----------------------------------------------------------------------------
# `Valid`Transformations
# -----------------------------------------------------------------------------
# Define Valid Transformations here
VALID_TRANSFORMATIONS: list = []


def get_transformations() -> Dict[str, Compose]:
    " returns a dictionary containing transformations for train & valid data"
    train_tfms = TRAIN_TRANSFORMATIONS + DEFAULT_TRANSFORMS
    valid_tfms = VALID_TRANSFORMATIONS + DEFAULT_TRANSFORMS
    # Compose Transformations
    train_tfms = A.Compose(
        train_tfms, bbox_params=A.BboxParams(BBOX_PARAMS, LABEL_FIELDS)
    )
    valid_tfms = A.Compose(
        valid_tfms, bbox_params=A.BboxParams(BBOX_PARAMS, LABEL_FIELDS)
    )
    tfms = {"train": train_tfms, "valid": valid_tfms}
    return tfms
