"""Training CLI."""
import json
from typing import Dict, List, Optional, Union

from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger


class TrainingCLI:
    def train_from_built(
        self, trainer: Trainer, model: LightningModule, datamodule: LightningDataModule,
    ):
        return trainer.fit(model=model, datamodule=datamodule)

    def build_trainer(
        self,
        trainer_params: Dict,
        callback_list: List[Callback],
        logger_dict: Dict,
        checkpoint_dict: Dict,
        device: Optional[Union[str, int]] = None,
    ) -> Trainer:
        if isinstance(device, str):
            device: Union[str, List[int]] = json.loads(device)
            if not isinstance(device, (str, list)):
                raise ValueError(f"Device is not valid: {device}")
        trainer_params["gpus"] = device
        logger: WandbLogger = WandbLogger(**logger_dict)
        model_checkpointer: ModelCheckpoint = ModelCheckpoint(**checkpoint_dict)
        callback_list += model_checkpointer
        return Trainer(callbacks=callback_list, logger=logger, *trainer_params)

    def train(
        self,
        model: LightningModule,
        datamodule: LightningDataModule,
        trainer_params: Dict,
        callback_list: List[Callback],
        logger_dict: Dict,
        checkpoint_dict: Dict,
        device: Optional[Union[str, int]] = None,
    ):
        trainer: Trainer = self.build_trainer(
            trainer_params=trainer_params,
            callback_list=callback_list,
            logger_dict=logger_dict,
            checkpoint_dict=checkpoint_dict,
            device=device,
        )
        return self.train_from_built(
            trainer=trainer, model=model, datamodule=datamodule,
        )
