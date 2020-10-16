import datetime

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from retinanet import Retinanet
from utils import _get_logger, collate_fn, load_obj

# COCO Utils
from utils.coco import CocoEvaluator, get_coco, get_coco_api_from_dataset
from utils.coco.coco_transforms import Compose, RandomHorizontalFlip, ToTensor

# PASCAL UTILS
from utils.pascal import get_pascal, PascalDataset
from utils.pascal.pascal_transforms import compose_transforms


def _get_model(hparams: DictConfig, **kwargs):
    model = Retinanet(**hparams.model, **kwargs)
    return model


class RetinaNetModel(pl.LightningModule):
    """
    Lightning Class to wrap the RetinaNet Model.
    So that it can be trainer with LightningTrainer.
    
    Args:
      haprams (`DictConfig`) : A `DictConfig` that stores the configs for training .
    """

    def __init__(self, hparams: DictConfig):
        super(RetinaNetModel, self).__init__()
        self.hparams = hparams
        # load model using model hparams
        self.fancy_logger = _get_logger(__name__)
        self.model = _get_model(self.hparams)

    def forward(self, xb, *args, **kwargs):
        return self.model(xb)

    def prepare_data(self):
        """
        load in the transformation & reads in the data from given paths.
        """
        data_params = self.hparams.dataset
        self.fancy_logger.info(f"DATASET_KIND : {data_params.kind}")

        # if coco-dataset format load in the coco datasets
        if data_params.kind == "coco":
            trn_tfms = Compose([ToTensor(), RandomHorizontalFlip(prob=0.5)])
            val_tfms = Compose([ToTensor()])
            # Load in the COCO Train and Test datasets
            # Coco doensn't have separate validation dataset
            # it only has the Train/Test dataset, so load in the train and test datasets
            self.trn_ds = get_coco(
                root=data_params.root_dir, image_set="train", transforms=trn_tfms
            )
            self.val_ds = None
            self.test_ds = get_coco(
                root=data_params.root_dir, image_set="val", transforms=val_tfms
            )

        # if pascal-dataset format load in the pascal dataset
        elif data_params.kind == "pascal":
            # instantiate the transformas for the dataset
            trn_tfms = [
                load_obj(i["class_name"])(**i["params"])
                for i in self.hparams.transforms
            ]
            trn_tfms = compose_transforms(trn_tfms)
            test_tfms = compose_transforms()
            # Load in the pascal dataset
            self.trn_ds = get_pascal(
                data_params.trn_paths[0],
                data_params.trn_paths[1],
                "train",
                transforms=trn_tfms,
            )

            # if validation dataset is given load in the validation dataset
            # else no validation dataset
            if data_params.valid_paths:
                self.val_ds = get_pascal(
                    data_params.valid_paths[0],
                    data_params.valid_paths[1],
                    "test",
                    transforms=test_tfms,
                )
            else:
                self.val_ds = None

            # Load in the test dataset
            self.test_ds = get_pascal(
                data_params.test_paths[0],
                data_params.test_paths[1],
                "test",
                transforms=test_tfms,
            )

        # if csv-data format
        elif data_params.kind == "csv":
            # instantiate the transformations
            trn_tfms = [
                load_obj(i["class_name"])(**i["params"])
                for i in self.hparams.transforms
            ]
            trn_tfms = compose_transforms(trn_tfms)
            test_tfms = compose_transforms()
            # load in the datasets from the csv files
            # load in the train dataset
            self.trn_ds = PascalDataset(data_params.trn_paths, trn_tfms)
            # load in the validation dataset if validation dataset is given
            if data_params.valid_paths:
                self.val_ds = PascalDataset(data_params.valid_paths, test_tfms)
            else:
                self.val_ds = None

            # Load in the test dataset
            self.test_ds = PascalDataset(data_params.test_paths, test_tfms)

        else:
            # raise error is dataset is other than "coco", "pascal", "csv"
            raise ValueError("DATASET_KIND not supported")

    def configure_optimizers(self, *args, **kwargs):
        # trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        # instantiate the Optimizer
        opt = self.hparams.optimizer.class_name
        self.optimizer = load_obj(opt)(params, **self.hparams.optimizer.params)
        # Log information about the Optimizer
        self.fancy_logger.info(f"OPTIMIZER_NAME : {self.optimizer.__class__.__name__}")
        self.fancy_logger.info(f"LEARNING_RATE  : {self.hparams.optimizer.params.lr}")
        self.fancy_logger.info(
            f"WEIGHT_DECAY   : {self.hparams.optimizer.params.weight_decay}"
        )

        # instantite schceduler if hprams.scheduler is not None
        if self.hparams.scheduler.class_name is None:
            # return only the optimizer if scheduler class_name is None
            return [self.optimizer]

        # else load in the scheduler
        else:
            sched = self.hparams.scheduler.class_name
            self.scheduler = load_obj(sched)(
                self.optimizer, **self.hparams.scheduler.params
            )
            # convert the scheduler to lightning format

            if not self.hparams.scheduler.monitor:
                self.scheduler = {
                    "scheduler": self.scheduler,
                    "interval": self.hparams.scheduler.interval,
                    "frequency": self.hparams.scheduler.frequency,
                }
            else:
                self.scheduler = {
                    "scheduler": self.scheduler,
                    "interval": self.hparams.scheduler.interval,
                    "frequency": self.hparams.scheduler.frequency,
                    "monitor": self.hparams.scheduler.monitor,
                }

            self.fancy_logger.info(
                f"LR_SCHEDULER_NAME : {self.scheduler['scheduler'].__class__.__name__}"
            )

            return [self.optimizer], [self.scheduler]

    def train_dataloader(self, *args, **kwargs):
        bs = self.hparams.dataloader.train_bs
        # instantiate train dataloader
        loader = DataLoader(
            self.trn_ds,
            bs,
            shuffle=True,
            collate_fn=collate_fn,
            **self.hparams.dataloader.args,
        )
        return loader

    def val_dataloader(self, *args, **kwargs):
        # if validation dataset is None
        # return None as dataloader
        if self.val_ds is None:
            return None

        else:
            bs = self.hparams.dataloader.valid_bs
            # instantiate validation dataloader
            loader = DataLoader(
                self.val_ds, bs, collate_fn=collate_fn, **self.hparams.dataloader.args
            )
            return loader

    def test_dataloader(self, *args, **kwargs):
        bs = self.hparams.dataloader.test_bs
        loader = DataLoader(
            self.test_ds, bs, collate_fn=collate_fn, **self.hparams.dataloader.args
        )
        # instantiate coco_api to track metrics
        prompt = "Converting dataset annotations in 'test_dataset' to COCO format for inference ..."
        self.fancy_logger.info(prompt)
        # convert test dataset to coco-format
        # to evaluate using COCO-API
        coco = get_coco_api_from_dataset(loader.dataset)
        self.test_evaluator = CocoEvaluator(coco, ["bbox"])
        prompt = f"Conversion finished, num images: {loader.dataset.__len__()}"
        self.fancy_logger.info(prompt)
        return loader

    # Training step :
    # model gives out the regression_loss(smooth_l1_loss) and
    # classification_loss(focal_Loss) of the model inputs ie., the train dataloader
    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, targets, _ = batch  # unpack the one batch from the DataLoader
        targets = [{k: v for k, v in t.items()} for t in targets]  # Unpack the Targets
        # Calculate Losses {regression_loss , classification_loss}
        loss_dict = self.model(images, targets)
        # Calculate Total Loss
        losses = sum(loss for loss in loss_dict.values())
        return {"loss": losses, "log": loss_dict, "progress_bar": loss_dict}

    # validaton step:
    # model gives out the regression_loss(smooth_l1_loss) and
    # classification_loss(focal_Loss) of the model inputs ie., the validation dataloader
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

    # test step :
    # the model computes the AP results on using COCO-API
    # with the predictions of the model
    # to get the predictions of the model we need to use the .predict method
    # of the retinanet model
    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model.predict(images)
        res = {t["image_id"].item(): o for t, o in zip(targets, outputs)}
        self.test_evaluator.update(res)
        return {}

    def test_epoch_end(self, outputs, *args, **kwargs):
        # coco results
        self.fancy_logger.info("Preparing results for COCO format ...")
        self.fancy_logger.info("Evaluating predictions ...")
        # compute COCO-API results
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


class LogCallback(pl.Callback):
    """
    Callback to handle logging within pl_module
    """

    def on_fit_start(self, trainer, pl_module):
        eps = trainer.max_epochs
        pl_module.fancy_logger.info(f"MAX_EPOCHS : {eps}")

    def on_train_start(self, trainer, pl_module):
        self.train_start = datetime.datetime.now().replace(microsecond=0)
        prompt = f"Training on {pl_module.train_dataloader().dataset.__len__()} images"
        pl_module.fancy_logger.info(prompt)
        prompt = f"Training from iteration {trainer.global_step} : "
        pl_module.fancy_logger.info(prompt)

    def on_train_end(self, trainer, pl_module):
        self.train_end = datetime.datetime.now().replace(microsecond=0)
        prompt = f" Total compute time : {self.train_end - self.train_start}"
        pl_module.fancy_logger.info(prompt)

    def on_test_start(self, trainer, pl_module):
        self.test_start = datetime.datetime.now().replace(microsecond=0)
        prompt = (
            f"Start Inference on {pl_module.test_dataloader().dataset.__len__()} images"
        )
        pl_module.fancy_logger.info(prompt)

    def on_test_end(self, trainer, pl_module):
        self.test_end = datetime.datetime.now().replace(microsecond=0)
        prompt = f" Total inference time : {self.test_end - self.test_start}"
        pl_module.fancy_logger.info(prompt)
