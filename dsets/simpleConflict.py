import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *

REMOTE_ROOT = f"{REMOTE_ROOT_URL}/data/dsets"


class McfConflictDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            multi: bool = False,
            size: typing.Optional[int] = None,
            exp_type: str = "A+B",
            *args,
            **kwargs,
    ):
        data_dir = Path(data_dir)
        cf_loc = data_dir / (
            "multi_counterfact.json"
        )
        cf_conflict_loc = data_dir / (
            "multi-counterfact-conflict.json"
        )
        if not cf_loc.exists():
            print("No mcf data")
        if not cf_conflict_loc.exists():
            print("No mcf_conflict data")
        with open(cf_loc, "r") as f:
            ori_data = json.load(f)
        if size is not None:
            ori_data = ori_data[:size]
        with open(cf_conflict_loc, "r") as f:
            conflict_data = json.load(f)
        if size is not None:
            conflict_data = conflict_data[:size]
        if exp_type == "A+B":
            ori_data.extend(conflict_data)
            self.data = ori_data
        elif exp_type == "_A":
            self.data = ori_data
        elif exp_type == "_B":
            self.data = conflict_data
        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

# class MultiCounterFactDataset(CounterFactDataset):
#     def __init__(
#         self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs
#     ):
#         super().__init__(data_dir, *args, multi=True, size=size, **kwargs)
