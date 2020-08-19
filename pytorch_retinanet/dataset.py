from typing import Dict
import cv2
import pandas as pd
import torch
from torch import nn
from torch.functional import Tensor
from torch.utils.data import DataLoader, Dataset

from . import config as cfg


class CSVDataset(Dataset):
    def __init__(self, trn: bool = True) -> None:
        # Path to the DataFrmae
        directory = cfg.CSV_DIR

        # Read in the DataFrame
        self.df = pd.read_csv(directory)

        # Get the List of all the unique images in the DataFrame
        self.image_ids = self.df[cfg.IMG_HEADER].unique()

        # Instantiate Transformations
        if trn:
            self.tfms = cfg.TRANSFORMATIONS['train_transforms']
        else:
            self.tfms = cfg.TRANSFORMATIONS['valid_transforms']

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem(self, idx):
        # Grab the image, bbox & labels from the DataFrame
        image_id = self.image_ids[idx]
        im = cv2.cvtColor(cv2.imread(image_id), cv2.COLOR_BGR2RGB)

        # extract the bounding boxes
        records = self.df[self.df[cfg.IMG_HEADER] == image_id]

        boxes = (
            records[
                [cfg.XMIN_HEADER,
                 cfg.YMIN_HEADER,
                 cfg.YMIN_HEADER,
                 cfg.YMAX_HEADER]
            ].values)

        # claculate bbox area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # Grab the class Labels
        class_labels = records[cfg.CLASS_HEADER].values

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
