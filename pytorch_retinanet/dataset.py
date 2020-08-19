from typing import *

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, data
from .anchors import ifnone

from . import config as cfg


class CSVDataset(Dataset):
    """
    Returns a `torch.utils.data.Dataset` instance from `csv` values.

    Each `CSV` file should contain 
           - filepath: path to the Input Image
           - xmin    : `xmin` values for the bounding boxes.
           - ymin    : `ymin` values for the bounding boxes.
           - xmax    : `xmax` values for the bounding boxes.
           - ymax    : `ymax` values for the bounding boxes.
    Each Items in the CSV file should correspond to a single annotation.

    Arguments:
    - trn(bool)                    : `Wheter` training data or `validation` data.
    - directory(str)               :  Path to `the csv` file.
    - tfms (Dict[str, Compose])    : `Albumentations` transforms to be applied to Image before feeding into the network.
                                      should be a dictionary where `train_transforms` & `val_transforms` should be the 
                                      transformations for the `training` or `validation data` respectively.
    - filepath (str)               :  `CSV` header for the Image File Paths.        
    - xmin_header (str)            :  `CSV` header for the xmin values for the `annotations`.
    - ymin_header (str)            :  `CSV` header for the xmin values for the `annotations`.
    - xmax_header (str)            :  `CSV` header for the xmin values for the `annotations`.
    - ymax_header (str)            :  `CSV` header for the xmin values for the `annotations`.
    - class_header (str)           :  `CSV` header for the class values for the `annotations`. Each labels shuld be an Integer.

    >>> See `config.py` for default values.
    """

    def __init__(self,
                 trn: bool = True,
                 directory: str = cfg.CSV_DIR,
                 tfms: Dict = cfg.TRANSFORMATIONS,
                 filepath: str = cfg.IMG_HEADER,
                 xmin_header: str = cfg.XMIN_HEADER,
                 ymin_header: str = cfg.YMIN_HEADER,
                 xmax_header: str = cfg.XMAX_HEADER,
                 ymax_header: str = cfg.YMAX_HEADER,
                 class_header: str = cfg.CLASS_HEADER
                 ) -> None:

        # Read in the DataFrame
        self.df = pd.read_csv(directory)

        # Get the List of all the unique images in the DataFrame
        self.image_ids = self.df[filepath].unique()

        # Instantiate Transformations
        if trn:
            self.tfms = tfms['train_transforms']
        else:
            self.tfms = tfms['valid_transforms']

        self.bbox_headers = [
            xmin_header, ymin_header, xmax_header, ymax_header
        ]
        self.cls_header = class_header

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        # Grab the image, bbox & labels from the DataFrame
        image_id = self.image_ids[idx]
        im = cv2.cvtColor(cv2.imread(image_id), cv2.COLOR_BGR2RGB)

        # extract the bounding boxes
        records = self.df[self.df[cfg.IMG_HEADER] == image_id]

        boxes = (
            records[
                [
                    self.bbox_headers[0],
                    self.bbox_headers[1],
                    self.bbox_headers[2],
                    self.bbox_headers[3],
                ]
            ].values)

        # claculate bbox area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # Grab the class Labels
        class_labels = records[self.cls_header].values.tolist()

        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        # apply transformations
        transformed = self.tfms(
            image=im,
            bboxes=boxes,
            class_labels=class_labels
        )

        # Grab the `transformed` `image`, `bboxes`, `class_labels`
        image = transformed['image']
        boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
        class_labels = torch.as_tensor(
            transformed['class_labels'], dtype=torch.int32)

        # Create the `Target` Dictionary
        target = {}
        target['image_id'] = torch.tensor([idx])
        target['image_id'] = torch.tensor([idx])
        target['labels'] = class_labels
        target['area'] = area
        target['iscrowd'] = iscrowd

        return image.float(), target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(
        dataset: Dataset = None,
        trn: Optional[bool] = False,
        batch_size: int = cfg.BATCH_SIZE,
        shuffle: bool = cfg.SHUFFLE,
        pin_memory: bool = cfg.PIN_MEMORY,
        num_workers: int = cfg.NUM_WORKERS,
        drop_last: bool = cfg.DROP_LAST,
        ** kwargs) -> DataLoader:
    '''
    Returns a PyTorch `DataLoader Instance`.
    '''
    dataset = ifnone(dataset, CSVDataset(trn))
    loader = (
        DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=drop_last,
            **kwargs))

    return loader
