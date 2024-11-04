import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

class RoundEditDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        conflict_dir: str,
        whetherConflict = False,
        size: typing.Optional[int] = None,
        *args,
        **kwargs,
    ):
        with open(conflict_dir) as fp:
            # self.data = json.load(fp)
            self.data = json.load(fp)
        if size is not None:
            self.data = self.data[:size]
        if whetherConflict:
            with open(conflict_dir) as fp:
                conflict_data = json.load(fp)
                self.data.extend(conflict_data)
        print(f"Loaded RoundEdit with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
