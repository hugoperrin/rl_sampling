"""Denoiser DataModule."""
import os
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from rl_sampling.data.denoiser.denoiser_dataset import DenoiserDataset


class DenoiserDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = os.path.join("data", "denoiser"),
        batch_size: int = 32,
        split_perc: float = 0.8,
        seed: int = 0,
    ):
        super().__init__()
        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.split_perc: float = split_perc
        self.seed: int = seed

    def setup(self, stage: Optional[str] = None):
        full_dataset = DenoiserDataset()
        train_split: int = int(self.split_perc * len(full_dataset))
        val_split: int = int((1 - self.split_perc) * len(full_dataset)) // 2
        test_split: int = len(full_dataset) - train_split - val_split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_split, val_split, test_split],
            generator=torch.manual_seed(self.seed),
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
