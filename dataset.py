from typing import *

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config as cfg
from src.anchors import ifnone
from transforms import get_transformations


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
    1. trn          (bool)             :  Wether training data or `validation` data.
    2. directory    (str)              :  Path to `the csv` file.
    3. filepath     (str)              : `CSV` header for the Image File Paths.        
    4. xmin_header  (str)              : `CSV` header for the xmin values for the `annotations`.
    5. ymin_header  (str)              : `CSV` header for the xmin values for the `annotations`.
    6. xmax_header  (str)              : `CSV` header for the xmin values for the `annotations`.
    7. ymax_header  (str)              : `CSV` header for the xmin values for the `annotations`.
    8. class_header (str)              : `CSV` header for the class values (integer) for the `annotations`.
    9. transformations (Dict[Compose]) : A dictionary contraining albumentation transforms for `train` & `valid`

    >>> See `config.py` for default values.
    """

    def __init__(
        self,
        trn: bool,
        directory: Optional[str] = None,
        filepath: Optional[str] = None,
        xmin_header: Optional[str] = None,
        ymin_header: Optional[str] = None,
        xmax_header: Optional[str] = None,
        ymax_header: Optional[str] = None,
        class_header: Optional[str] = None,
        transformations: Optional[Dict] = None,
    ) -> None:

        # Unpack flags
        self.is_trn = trn

        if self.is_trn:
            self.csv_pth = ifnone(directory, cfg.TRAIN_CSV_DIR)
        else:
            self.csv_pth = ifnone(directory, cfg.VAL_CSV_DIR)

        self.file_header = ifnone(filepath, cfg.IMG_HEADER)
        self.xmin_header = ifnone(xmin_header, cfg.XMIN_HEADER)
        self.ymin_header = ifnone(ymin_header, cfg.YMIN_HEADER)
        self.xmax_header = ifnone(xmax_header, cfg.XMAX_HEADER)
        self.ymax_header = ifnone(ymax_header, cfg.YMAX_HEADER)
        self.cls_header = ifnone(class_header, class_header)
        self.tfms_dict = ifnone(transformations, get_transformations())

        self.df = pd.read_csv(self.csv_pth)
        self.image_ids = self.df[self.file_header]

        try:
            if self.is_trn:
                self.tfms = self.tfms_dict["train"]
            else:
                self.tfms = self.tfms_dict["valid"]
        except ValueError:
            print("Invalid `transformations` check `transforms.py`")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index):
        # Grab the Image
        image_idx = self.image_ids[index]
        image = cv2.cvtColor(cv2.imread(image_idx), cv2.COLOR_BGR2RGB)

        records = self.df[self.df[self.file_header] == image_idx]

        boxes = records[
            [self.xmin_header, self.ymin_header, self.xmax_header, self.ymax_header]
        ].values()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        class_labels = records["target"].values.tolist()

        # apply transformations
        transformed = self.tfms(image=image, bboxes=boxes, class_labels=class_labels)
        image = transformed["image"]
        boxes = transformed["bboxes"]
        class_labels = transformed["class_labels"]

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


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(
    dataset: Optional[Dataset] = None, train: Optional[bool] = None,
) -> DataLoader:
    """
    Returns a `PyTorch` DataLoader Instance for given `dataset`
    If `Dataset` is None Dataset defaults to `CSVDataset`.
    If `CSVDataset` then `train`(bool) must be given. Dataset will be loaded 
    from the `Dataset Flags` given in config.py
    """
    if dataset is None:
        assert train is not None, "if `dataset` is not given `train` must be specified"

    dataset = ifnone(dataset, CSVDataset(trn=train))
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=cfg.SHUFFLE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=cfg.DROP_LAST,
    )
    return dataloader
