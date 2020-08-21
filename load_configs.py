import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf/config.yaml")
def get_config(cfg: DictConfig) -> None:
    print(cfg.pretty())
