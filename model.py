import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from retinanet import Retinanet
from utils import collate_fn, load_obj
from utils.coco import CocoEvaluator, get_coco, get_coco_api_from_dataset
from utils.coco.coco_transforms import Compose, RandomHorizontalFlip, ToTensor
from utils.pascal import get_pascal, PascalDataset
from utils.pascal.pascal_transforms import compose_transforms
from typing import Union
import argparse


class RetinaNetModel(pl.LightningModule):
    """
    Lightning Class to wrap the RetinaNet Model.
    So that it can be trainer with LightningTrainer.
    
    Args:
      haprams (`DictConfig`) : A `DictConfig` that stores the configs for training .
    """

    def __init__(self, hparams: Union[DictConfig, argparse.Namespace]):
        super(RetinaNetModel, self).__init__()
        self.hparams = hparams
        self.net = Retinanet(**hparams.model)

    def forward(self, xb, *args, **kwargs):
        output = self.net(xb)
        return output

    def prepare_data(self):
        """
        load in the transformation & reads in the data from given paths.
        """
        data_params = self.hparams.dataset

        if data_params.kind == "coco":
            trn_tfms = Compose([ToTensor(), RandomHorizontalFlip(prob=0.5)])
            val_tfms = Compose([ToTensor()])

            self.trn_ds = get_coco(
                root=data_params.root_dir, image_set="train", transforms=trn_tfms
            )
            self.val_ds = None
            self.test_ds = get_coco(
                root=data_params.root_dir, image_set="val", transforms=val_tfms
            )

        elif data_params.kind == "pascal":
            trn_tfms = [
                load_obj(i["class_name"])(**i["params"])
                for i in self.hparams.transforms
            ]
            trn_tfms = compose_transforms(trn_tfms)
            test_tfms = compose_transforms()

            self.trn_ds = get_pascal(
                data_params.trn_paths[0],
                data_params.trn_paths[1],
                "train",
                transforms=trn_tfms,
            )

            if data_params.valid_paths:
                self.val_ds = get_pascal(
                    data_params.valid_paths[0],
                    data_params.valid_paths[1],
                    "test",
                    transforms=test_tfms,
                )
            else:
                self.val_ds = None

            self.test_ds = get_pascal(
                data_params.test_paths[0],
                data_params.test_paths[1],
                "test",
                transforms=test_tfms,
            )

        elif data_params.kind == "csv":
            trn_tfms = [
                load_obj(i["class_name"])(**i["params"])
                for i in self.hparams.transforms
            ]
            trn_tfms = compose_transforms(trn_tfms)
            test_tfms = compose_transforms()
            self.trn_ds = PascalDataset(data_params.trn_paths, trn_tfms)
            if data_params.valid_paths:
                self.val_ds = PascalDataset(data_params.valid_paths, test_tfms)
            else:
                self.val_ds = None
            self.test_ds = PascalDataset(data_params.test_paths, test_tfms)

        else:
            raise ValueError("DATASET_KIND not supported")

    def configure_optimizers(self, *args, **kwargs):
        opt = self.hparams.optimizer.class_name
        self.optimizer = load_obj(opt)(
            self.net.parameters(), **self.hparams.optimizer.params
        )

        if self.hparams.scheduler.class_name is None:
            return [self.optimizer]

        else:
            schedps = self.hparams.scheduler
            __scheduler = load_obj(schedps.class_name)(self.optimizer, **schedps.params)

            if not self.hparams.scheduler.monitor:
                self.scheduler = {
                    "scheduler": __scheduler,
                    "interval": schedps.interval,
                    "frequency": schedps.frequency,
                }
            else:
                self.scheduler = {
                    "scheduler": __scheduler,
                    "interval": schedps.interval,
                    "frequency": schedps.frequency,
                    "monitor": schedps.scheduler.monitor,
                }

            return [self.optimizer], [self.scheduler]

    def train_dataloader(self, *args, **kwargs):
        bs = self.hparams.dataloader.train_bs
        loader = DataLoader(
            self.trn_ds,
            bs,
            shuffle=True,
            collate_fn=collate_fn,
            **self.hparams.dataloader.args,
        )
        return loader

    def val_dataloader(self, *args, **kwargs):
        if self.val_ds is None:
            return None
        else:
            bs = self.hparams.dataloader.valid_bs
            loader = DataLoader(
                self.val_ds, bs, collate_fn=collate_fn, **self.hparams.dataloader.args
            )
            return loader

    def test_dataloader(self, *args, **kwargs):
        bs = self.hparams.dataloader.test_bs
        loader = DataLoader(
            self.test_ds, bs, collate_fn=collate_fn, **self.hparams.dataloader.args
        )
        coco = get_coco_api_from_dataset(loader.dataset)
        self.test_evaluator = CocoEvaluator(coco, ["bbox"])
        return loader

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, targets, _ = batch  # unpack the one batch from the DataLoader
        targets = [{k: v for k, v in t.items()} for t in targets]  # Unpack the Targets
        # Calculate Losses {regression_loss , classification_loss}
        loss_dict = self.model(images, targets)
        # Calculate Total Loss
        losses = sum(loss for loss in loss_dict.values())
        return {"loss": losses, "log": loss_dict, "progress_bar": loss_dict}

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, targets, _ = batch  # unpack the one batch from the DataLoader
        targets = [{k: v for k, v in t.items()} for t in targets]  # Unpack the Targets
        # Calculate Losses {regression_loss , classification_loss}
        loss_dict = self.model(images, targets)
        # Calculate Total Loss
        loss = sum(loss for loss in loss_dict.values())
        loss = torch.as_tensor(loss)
        logs = {"val_loss": loss}
        return {
            "val_loss": loss,
            "log": logs,
            "progress_bar": logs,
        }

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model.predict(images)
        res = {t["image_id"].item(): o for t, o in zip(targets, outputs)}
        self.test_evaluator.update(res)
        return {}

    def test_epoch_end(self, outputs, *args, **kwargs):
        self.test_evaluator.accumulate()
        self.test_evaluator.summarize()
        metric = self.test_evaluator.coco_eval["bbox"].stats[0]
        metric = torch.as_tensor(metric)
        logs = {"AP": metric}
        return {
            "AP": metric,
            "log": logs,
            "progress_bar": logs,
        }

