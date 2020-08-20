import argparse
from src.dataset import get_dataloader
from src.models import Retinanet
from src.trainer import DefaultTrainer


parser = argparse.ArgumentParser()
parser.add_argument(
    "-lr", "--learning_rate", type=float, default=1e-03, help="optimizer learning rate"
)
parser.add_argument("-mom", "--momentum", type=float, default=0.9, help="momentum")
parser.add_argument(
    "-wd", "--weight_decay", type=float, default=1e-02, help="optimizer weight decay"
)
parser.add_argument("-e", "--epochs", type=int, default=30, help="# epochs")
