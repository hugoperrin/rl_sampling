"""Abstract class for general models."""

from abc import abstractclassmethod
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger
from pytorch_lightning import LightningModule

from _typeshed import NoneType
from rl_sampling.utils.types import TensorLike


class BaseModel(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.init_loss(*args, **kwargs)
        self.init_model(*args, **kwargs)
        self.init_metrics(*args, **kwargs)
        self.init_utils(*args, **kwargs)

    def init_loss(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(
            "You must specify a loss in init loss in order to train: inference only"
        )

    def init_utils(self, *args: Any, **kwargs: Any) -> None:
        self.step_counters: Dict[str, int] = {"training": 0, "validation": 0}
        self.epoch_counters: Dict[str, int] = {"training": 0, "validation": 0}

    def init_model(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("You must specify a model in init model")

    def init_metrics(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(
            "You must specify metrics in init metrics so as to track training and validation metrics"
        )
        self.metrics: torch.nn.ModuleDict = torch.ModuleDict(
            {"training": {}, "validation": {}}
        )

    @property
    def loss(self):
        return self.__loss

    @abstractclassmethod
    def forward(self, x: TensorLike) -> TensorLike:
        ...

    def training_step(self, batch: TensorLike, batch_idx: int):

        inputs: TensorLike
        targets: TensorLike
        inputs, targets = self.extract_batch_data(batch=batch, batch_idx=batch_idx,)
        predictions: TensorLike = self.get_predictions(
            inputs=inputs, mode="training",
        )
        loss = self.get_loss_value(
            inputs=inputs, targets=targets, predictions=predictions, mode="training",
        )
        self.logging_metrics(
            inputs=inputs,
            targets=targets,
            predictions=predictions,
            batch_idx=batch_idx,
            mode="training",
        )
        return {"loss": loss}

    def extract_batch_data(
        self, batch: TensorLike, batch_idx: int
    ) -> Tuple[TensorLike, TensorLike]:
        inputs, targets = batch
        return inputs, targets

    def get_predictions(self, inputs: TensorLike, mode: str) -> TensorLike:
        self.step_counters[mode] += 1
        return self.forward(inputs)

    def get_loss_value(
        self,
        inputs: TensorLike,
        targets: TensorLike,
        predictions: TensorLike,
        mode: str,
    ) -> TensorLike:
        return self.loss(predictions, targets)

    def logging_metrics(
        self,
        inputs: Optional[TensorLike],
        targets: Optional[TensorLike],
        predictions: Optional[TensorLike],
        mode: str,
        batch_idx: Optional[int] = None,
        is_step: bool = True,
    ) -> None:
        metrics_logged: Dict[str, float] = {}
        for metric_name, metric in self.metrics[mode].items():
            metric_val: float
            name: str
            if is_step:
                metric_val = metric(predictions, targets)
                name = f"{mode}/{metric_name}_step"
            else:
                metric_val = metric.compute()
                name = f"{mode}/{metric_name}_epoch"
            metrics_logged[name] = metric_val
        self.log_dict(metrics_logged)

    def training_epoch_end(self, training_step_outputs):
        self.logging_metrics(
            inputs=None, targets=None, predictions=None, mode="training", is_step=False,
        )

    def validation_epoch_end(self, training_step_outputs):
        self.logging_metrics(
            inputs=None,
            targets=None,
            predictions=None,
            mode="validation",
            is_step=False,
        )

    def configure_optimizers(self):
        logger.warning("No optimizer is specified")
        optimizer: NoneType = None
        return optimizer
