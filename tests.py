import hydra
from omegaconf import DictConfig, OmegaConf
import albumentations as A
from src.utilities import load_obj


def load_transformations(cfg: DictConfig):
    train_transforms = [
        load_obj(i["class_name"])(**i["params"])
        for i in cfg["augmentation"]["train"]["augs"]
    ]
    bbox_params = OmegaConf.to_container((cfg["augmentation"]["train"]["bbox_params"]))
    transformations = A.Compose(train_transforms, bbox_params=bbox_params)
    return transformations


@hydra.main(config_path="config/config.yaml")
def get_config(cfg: DictConfig) -> None:
    tfms = load_transformations(cfg)
    print(tfms)


# from src.models import Retinanet

# if __name__ == "__main__":
#     import torch

#     m = Retinanet()
#     m.eval()
#     out = m([torch.randn(3, 355, 355), torch.randn(3, 355, 355)])
#     print(m)
#     print(out)
