"""Abstract class for regression models."""

from typing import Any
import torch

from torchmetrics.regression.mean_absolute_error import MeanAbsoluteError
from torchmetrics.regression.ssim import SSIM
from torchmetrics.regression.mean_squared_error import MeanSquaredError
from rl_sampling.models.base_model import BaseModel


class RegressionModel(BaseModel):
    def init_metrics(self, *args: Any, **kwargs: Any) -> None:
        self.metrics: torch.nn.ModuleDict = torch.ModuleDict(
            {
                "training": {
                    "mse": MeanSquaredError(),
                    "mae": MeanAbsoluteError(),
                    "ssim": SSIM(),
                },
                "validation": {
                    "mse": MeanSquaredError(),
                    "mae": MeanAbsoluteError(),
                    "ssim": SSIM(),
                },
            }
        )
