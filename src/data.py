from typing import *

import cv2
import pandas as pd
import torch
from albumentations.core.composition import Compose
from torch.utils.data import DataLoader, Dataset

from .utilities import collate_fn


class CSVDataset(Dataset):
    """
    Returns a `torch.utils.data.Dataset` instance from `csv` values.

    Each `CSV` file should contain
        * filepath: path to the Input Image
        * xmin    : `xmin` values for the bounding boxes.
        * ymin    : `ymin` values for the bounding boxes.
        * xmax    : `xmax` values for the bounding boxes.
        * ymax    : `ymax` values for the bounding boxes.

    Each Items in the CSV file should contain a single annotation.

    Arguments :
    ---------
    1. is_train         (bool)         :  Wether training data or `validation` data.
    2. tfms_dict    (Dict[Compose])    :  A dictionary contraining albumentation transforms for `train` & `valid`.
    3. directory    (str)              :  Path to `the csv` file.
    4. file_header    (str)            : `CSV` header for the Image File Paths.
    5. xmin_header  (str)              : `CSV` header for the xmin values for the `annotations`.
    6. ymin_header  (str)              : `CSV` header for the xmin values for the `annotations`.
    7. xmax_header  (str)              : `CSV` header for the xmin values for the `annotations`.
    8. ymax_header  (str)              : `CSV` header for the xmin values for the `annotations`.
    9. class_header (str)              : `CSV` header for the class values (integer) for the `annotations`.
    """

    def __init__(
        self,
        is_train: bool,
        tfms_dict: Dict[Compose],
        directory: str,
        file_header: str,
        xmin_header: str,
        ymin_header: str,
        xmax_header: str,
        ymax_header: str,
        class_header: str,
    ) -> None:

        # Unpack flags
        self.is_train = is_train
        self.csv_pth = directory
        self.file_header = file_header
        self.xmin_header = xmin_header
        self.ymin_header = ymin_header
        self.xmax_header = xmax_header
        self.ymax_header = ymax_header
        self.cls_header = class_header
        # Read in the DataFrmae & extract the unique ImageIds
        self.df = pd.read_csv(self.csv_pth)
        self.image_ids = self.df[self.file_header]

        # Instantiate train configs in self.is_train
        if self.is_trn:
            self.tfms = tfms_dict["train"]
        else:
            self.tfms = tfms_dict["valid"]

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.df

    @property
    def unique_files_idxs(self) -> List:
        return self.image_ids
    
    @property
    def transformations(self) -> None:
        return print(self.tfms)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index):
        # Grab the Image
        image_idx = self.image_ids[index]
        image = cv2.cvtColor(cv2.imread(image_idx), cv2.COLOR_BGR2RGB)

        # Extract the records corresponding to the unique entry in the DataFrame
        records = self.df[self.df[self.file_header] == image_idx]
        # Grab the `boxes`, `class_labels` from the `records`
        boxes = records[
            [self.xmin_header, self.ymin_header, self.xmax_header, self.ymax_header]
        ].values()
        class_labels = records["target"].values.tolist()

        # calculate area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # apply transformations to the `image`, `bboxes`, `labels`
        transformed = self.tfms(image=image, bboxes=boxes, class_labels=class_labels)
        image = transformed["image"]
        boxes = transformed["bboxes"]
        class_labels = transformed["class_labels"]

        # Convert all to `Tensors`
        image_idx = torch.tensor([index])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        area = torch.tensor(area, dtype=torch.float32)
        class_labels = torch.tensor(class_labels, dtype=torch.int64)
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        # Create the Target Dictionary
        target = {}
        target["image_id"] = image_idx
        target["boxes"] = boxes
        target["labels"] = class_labels
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target, image_idx


def get_dataloader(dataset, **kwargs) -> DataLoader:
    """
    Returns a `PyTorch` DataLoader Instance for given `Dataset`

    Arguments:
    ----------
     1. dataset (Dataset): `A torch.utils.Dataset` instance.
     2. **kwargs         : Dataloader Flags
    """
    dataloader = DataLoader(dataset, collate_fn=collate_fn, **kwargs)
    return dataloader
