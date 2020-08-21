import importlib
from typing import *

from torch.utils.data import DataLoader, Dataset

from src.config import *
from src.dataset import CSVDataset


def collate_fn(batch):
    "`collate_fn` for pytorch obj_detection dataloader"
    return tuple(zip(*batch))


def ifnone(a: Any, b: Any) -> Any:
    """`a` if `a` is not None, otherwise `b`"""
    if a is not None:
        return a
    else:
        return b


def get_dataloader(
    dataset: Optional[Dataset] = None, train: Optional[bool] = None, **kwargs
) -> DataLoader:
    """
    Returns a `PyTorch` DataLoader Instance for given `dataset`

    Arguments:
    ----------
     1. dataset (Optional[Dataset]): `A torch.utils.Dataset` instance.
                If `dataset` is None Dataset defaults to `CSVDataset`.
                `CSV` dataset is loaded using default config flags given in `config.py`
     2. train   (Optional[bool])   : boolean wheter train or valid.
     3. **kwargs                   : Dataloader Flags
    """
    if dataset is None:
        assert train is not None, "if `dataset` is not given `train` must be specified"

    dataset = ifnone(dataset, CSVDataset(trn=train))
    dataloader = DataLoader(dataset, collate_fn=collate_fn, **kwargs)
    return dataloader


# https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)
