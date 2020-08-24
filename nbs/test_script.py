from src.models import Retinanet
from src.eval_utils.coco_eval import CocoEvaluator
from src.eval_utils.coco_utils import get_coco_api_from_dataset
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from sklearn import preprocessing, model_selection
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
import re
import xml.etree.ElementTree as ET
from pathlib import Path

pl.seed_everything(123)
warnings.filterwarnings("ignore")

img_desc = Path("/content/oxford-iiit-pet/images")
annot_dir = Path("/content/oxford-iiit-pet/annotations/xmls")
annots = list(annot_dir.iterdir())
annots = [str(a) for a in annots]


def xml_to_csv(pths):
    """Extracts the filenames and the bboxes from the xml_list"""
    xml_list = []
    for xml_file in pths:
        # Read in the xml file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for m in root.findall("object"):
            try:
                value = (
                    # Extract the path to the image
                    str(img_desc / root.find("filename").text),
                    # Extract the bounding boxes
                    # 1. xmin
                    float(m[4][0].text),
                    # 2. ymin
                    float(m[4][1].text),
                    # 3. xmax
                    float(m[4][2].text),
                    # 4. ymax
                    float(m[4][3].text),
                )
                xml_list.append(value)
            except:
                pass
    col_n = ["filename", "xmin", "ymin", "xmax", "ymax"]
    df = pd.DataFrame(xml_list, columns=col_n)
    return df


########################################################################################################################

################################################  Data Preprocessing    ################################################
df = xml_to_csv(annots)
pat = r"/([^/]+)_\d+.jpg$"
pat = re.compile(pat)

df["class"] = [pat.search(fname).group(1).lower() for fname in df.filename]

le = preprocessing.LabelEncoder()
df["target"] = le.fit_transform(df["class"].values) + 1

# Take a Small Subset
_, df = model_selection.train_test_split(df, stratify=df["target"], test_size=0.5)
df.reset_index(drop=True)

df = df.sample(frac=1).reset_index(drop=True)
df_train, df_test = model_selection.train_test_split(
    df, stratify=df["target"], test_size=0.25, shuffle=True
)
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
########################################################################################################################

#################################### Image Transformations   ###########################################################
transformations = [
    A.HorizontalFlip(p=0.5),
    A.CLAHE(),
    A.IAASharpen(),
    A.IAAPerspective(),
    A.OneOf([A.ShiftScaleRotate(), A.Rotate(limit=60),], p=1.0),
    A.OneOf([A.RandomShadow(), A.RandomBrightnessContrast(), A.Cutout()], p=0.5),
]

# Train Transformations
train_transformations = transformations + [
    A.ToFloat(max_value=255.0, always_apply=True),
    ToTensorV2(always_apply=True),
]

# Valid Transformations
valid_transformations = [
    A.ToFloat(max_value=255.0, always_apply=True),
    ToTensorV2(always_apply=True),
]

# Transformations:
transforms = {
    "train": A.Compose(
        train_transformations,
        p=1.0,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    ),
    "valid": A.Compose(
        valid_transformations,
        p=1.0,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    ),
}
########################################################################################################################

############################################  Utily Classes and Functions    ###########################################
def collate_fn(batch):
    return tuple(zip(*batch))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, train):
        self.df = dataframe
        self.image_ids = self.df["filename"]
        if train:
            self.tfms = transforms["train"]
        else:
            self.tfms = transforms["valid"]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Grab the Image
        image_id = self.image_ids[idx]
        im = cv2.cvtColor(cv2.imread(image_id), cv2.COLOR_BGR2RGB)
        # extract the bounding boxes
        records = self.df[self.df["filename"] == image_id]
        boxes = records[["xmin", "ymin", "xmax", "ymax"]].values
        # claculate area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        # Grab the Class Labels
        class_labels = records["target"].values.tolist()
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        # apply transformations
        transformed = self.tfms(image=im, bboxes=boxes, class_labels=class_labels)
        image = transformed["image"]
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        class_labels = torch.tensor(transformed["class_labels"])
        # target dictionary
        target = {}
        image_idx = torch.tensor([idx])
        target["image_id"] = image_idx
        target["boxes"] = boxes
        target["labels"] = class_labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        return image, target, image_idx


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        max_lr: float,
    ):

        super(LitModel, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.max_lr = max_lr

        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[16, 22], gamma=0.1
            ),
            "interval": "epoch",
            "frequency": 1,
        }

    def configure_optimizers(self, *args, **kwargs):
        optimizer = self.optimizer
        if self.scheduler is not None:
            scheduler = self.scheduler
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, *args, **kwargs
    ):
        # warm up lr
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.max_lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def train_dataloader(self, *args, **kwargs):
        return self.train_dl

    def val_dataloader(self, *args, **kwargs):
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
        losses = loss_dict["classification_loss"] + loss_dict["regression_loss"]
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


########################################################################################################################

###################################### Training Configurations ########################################################
def main(args):
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 12
    EPOCHS = args.epochs
    MAX_LR = args.lr
    NUM_CLASSES = (
        len(df["target"].unique()) + 1
    )  # len(df['target']).unique() classes + 1 background class
    print("------------------------")
    print("[INFO] ARGUMENTS: >>")
    print("[INFO] BACKBONE:", args.model)
    print("[INFO] EPOCHS:", args.epochs)
    print("[INFO] LEARNING_RATE:", args.lr)
    print("[INFO] WEIGHT_DECAY:", args.wd)
    print("[INFO] MOMENTUM:", args.mom)
    print("[INFO] PRECISION:", args.precision)
    print("------------------------")

    model = Retinanet(
        num_classes=NUM_CLASSES,
        backbone_kind=args.model,
        pretrained=True,
        freeze_bn=True,
    )

    train_ds = Dataset(df_train, train=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_ds = Dataset(df_test, train=False)
    val_dl = DataLoader(
        val_ds,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=MAX_LR,
        momentum=args.mom,
        weight_decay=args.wd,
    )

    ###################################### Lightning Modules ###############################################
    tb_logger = pl.loggers.TensorBoardLogger(save_dir="/content/logs")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        "content/saved_models", mode="max", monitor="bbox_IOU", save_top_k=-1
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        mode="max", monitor="bbox_IOU", patience=5
    )
    lightning_model = LitModel(model, optimizer, train_dl, val_dl, max_lr=MAX_LR)

    trainer = pl.Trainer(
        logger=[tb_logger],
        num_sanity_val_steps=0,
        early_stop_callback=early_stopping_callback,
        checkpoint_callback=checkpoint_callback,
        max_epochs=EPOCHS,
        precision=args.precision,
        gpus=1,
    )

    ################################################ Fit Model #############################################################
    trainer.fit(lightning_model)
    ########################################################################################################################


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=(0.02 / 8), type=float, help="learning_rate")
    parser.add_argument(
        "--wd", "--weight-decay", default=1e-4, type=float, help="weight_decay"
    )
    parser.add_argument("--epochs", "--epochs", default=26, type=int, help="num_epochs")
    parser.add_argument(
        "--model", "--model", default="resnet18", type=str, help="name_of_resnet_model"
    )
    parser.add_argument(
        "--precision", "--precision", default=16, type=int, help="precision"
    )
    parser.add_argument("--mom", "--momentum", default=0.9, type=float, help="momentum")
    args = parser.parse_args()

    main(args)
