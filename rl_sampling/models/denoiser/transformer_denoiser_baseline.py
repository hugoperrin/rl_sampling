"""Transformer denoiser baseline."""
import os
from typing import Dict, List, Optional, Union

import fire
from pytorch_lightning import Callback

from rl_sampling.models.regression_model import RegressionModel


class TransformerDenoiserBaseline(RegressionModel):
    ...


def train(device: Optional[Union[str, int]] = None):
    model = TransformerDenoiserBaseline()
    callbacks: List[Callback] = []
    trainer_params: Dict = {}
    logger_dict: Dict = {
        "name": "TransformerDenoiser",
        "experiment": "TransformerBaseline",
        "entity": "hugop",
    }
    checkpoint_dict: Dict = {
        "save_last": True,
        "save_top_k": 3,
        "save_weights_only": True,
        "mode": "min",
        "monitor": "validation/loss",
    }

    from rl_sampling.data.denoiser.denoiser_datamodule import DenoiserDataModule

    data_path: str = os.path.join("data", "denoiser")
    datamodule = DenoiserDataModule(
        data_dir=data_path, batch_size=16, train_split_perc=0.8,
    )

    from rl_sampling.utils.training_cli import TrainingCLI

    cli: TrainingCLI = TrainingCLI()
    return cli.train(
        model=model,
        datamodule=datamodule,
        trainer_params=trainer_params,
        callback_list=callbacks,
        logger_dict=logger_dict,
        checkpoint_dict=checkpoint_dict,
        device=device,
    )


if __name__ == "__main__":
    fire.Fire()
