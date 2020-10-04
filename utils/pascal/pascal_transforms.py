import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np


def compose_transforms(transforms=None):
    if transforms is None:
        transforms = []
    transforms.append(A.ToFloat(max_value=255.0, always_apply=True),)
    transforms.append(ToTensorV2(always_apply=True))

    c = A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    return c


def generate_pascal_category_names(df: pd.DataFrame):
    """
    Genearte a List contatining the Category names, 
    The index of the List will correspond to the integer
    value of the category name

    Args:
        df: pd.Dataframe: A pandas dataframe in the `pascal_voc_csv format`
    """
    cs = df["class"].unique()
    ls = df["labels"].unique()

    CATEGORY_NAMES = list(np.zeros(len(ls) + 1))
    for i, x in enumerate(ls):
        CATEGORY_NAMES[x] = cs[i]

    # Add the background class to the Category names
    # Since the labels start from the 1, we set the 0 value to be the
    # background class
    CATEGORY_NAMES[0] = "__background__"
    return CATEGORY_NAMES
