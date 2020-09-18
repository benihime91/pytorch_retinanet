from typing import Union

import albumentations as A
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    """
    Creates a object detection Dataset instance

    Args:
        1. dataframe : A pd.Dataframe instance or str corresponding to the 
                       path to the dataframe.
        For the Dataframe the `filename` column should correspond to the path to the images.
        Each row to should one annotations in the the form `xmin`, `ymin`, `xmax`, `yman`.
        Labels should be integers in the `labels` column.

        2. transforms: (A.Compose) transforms should be a albumentation transformations.
                        the bbox params should be set to `pascal_voc` & to pass in class
                        use `class_labels`                  
    """

    def __init__(
        self, dataframe: Union[pd.DataFrame, str], transforms: A.Compose,
    ) -> None:

        if isinstance(dataframe, str):
            dataframe = pd.read_csv(dataframe)

        self.tfms = transforms
        self.df = dataframe
        self.image_ids = self.df["filename"].unique()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        # Grab the Image
        image_id = self.image_ids[index]
        im = cv2.cvtColor(cv2.imread(image_id), cv2.COLOR_BGR2RGB)

        # extract the bounding boxes
        records = self.df[self.df["filename"] == image_id]
        boxes = records[["xmin", "ymin", "xmax", "ymax"]].values

        # claculate area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # Grab the Class Labels
        class_labels = records["labels"].values.tolist()

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        # apply transformations
        transformed = self.tfms(image=im, bboxes=boxes, class_labels=class_labels)
        image = transformed["image"]
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        class_labels = torch.tensor(transformed["class_labels"])

        # target dictionary
        target = {}
        image_idx = torch.tensor([index])
        target["image_id"] = image_idx
        target["boxes"] = boxes
        target["labels"] = class_labels
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target, image_idx
