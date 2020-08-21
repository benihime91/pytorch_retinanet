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
