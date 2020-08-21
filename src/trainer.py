from typing import *
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from .eval_utils.coco_eval import CocoEvaluator
from .eval_utils.coco_utils import get_coco_api_from_dataset


class DefaultTrainer(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_dl: DataLoader,
        val_dl: DataLoader,
        lr: float,
        scheduler: Dict[str, Any] = None,
    ) -> None:

        super(DefaultTrainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.learning_rate = lr
        self.train_dl = train_dl
        self.val_dl = val_dl

    def configure_optimizers(self, *args, **kwargs):
        optimizer = self.optimizer
        if self.scheduler is not None:
            scheduler = self.scheduler
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def optimizer_step(self, optimizer, *args, **kwargs):
        # warmup lr
        if self.trainer.global_step < 500:
            alpha = min(1.0, float(self.trainer.global_step + 1) / 500.0)
            for pg in self.optimizer.param_groups:
                pg["lr"] = alpha * self.max_lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def train_dataloader(self) -> DataLoader:
        return self.train_dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        valid_loader = self.val_dl

        # Prepare COCO Evaluator
        coco = get_coco_api_from_dataset(valid_loader.dataset)
        iou_types = ["bbox"]
        self.coco_evaluator = CocoEvaluator(coco, iou_types)
        return valid_loader

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        # Separate Losses
        loss_dict = self.model(images, targets)
        # Total Loss
        losses = sum(loss for loss in loss_dict.values())
        return {"loss": losses, "log": loss_dict, "progress_bar": loss_dict}

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images, targets)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        self.coco_evaluator.update(res)
        return {}

    def validation_epoch_end(self, outputs):
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        # coco main metric
        metric = self.coco_evaluator.coco_eval["bbox"].stats[0]
        metric = torch.as_tensor(metric)
        tensorboard_logs = {"bbox_IOU": metric}
        return {
            "val_loss": metric,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
