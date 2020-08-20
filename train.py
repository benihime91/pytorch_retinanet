import argparse
from src.dataset import get_dataloader
from src.models import Retinanet
from src.trainer import DefaultTrainer


model = Retinanet()
print(model)
