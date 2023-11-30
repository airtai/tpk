# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Any, Dict, List, Optional

import torch
from gluonts.core.component import validated
from gluonts.itertools import select
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR

from .module import TSMixerModel


class TSMixerLightningModule(LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``TSMixerModel`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``TSMixerModel`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model
        ``TSMixerModel`` to be trained.
    loss
        Loss function to be used for training,
        default: ``NegativeLogLikelihood()``.
    weight_decay
        Weight decay regularization parameter, default: ``1e-8``.
    patience
        Patience parameter for learning rate scheduler, default: ``10``.
    """

    @validated()  # type: ignore
    def __init__(
        self,
        model: TSMixerModel,
        epochs: int,
        steps_per_epoch: int,
        loss: Optional[DistributionLoss] = None,
        weight_decay: float = 1e-8,
        patience: int = 10,
    ) -> None:
        super().__init__()
        self.loss = NegativeLogLikelihood() if loss is None else loss
        self.save_hyperparameters()
        self.model = model
        self.weight_decay = weight_decay
        self.patience = patience
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = 0.0
        self.example_input_array = tuple(
            [
                torch.zeros(shape, dtype=self.model.input_types()[name])
                for (name, shape) in self.model.input_shapes().items()
            ]
        )

    def forward(self, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        return self.model(*args, **kwargs)

    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        # context = batch["past_target"]
        target = batch["future_target"]
        observed_target = batch["future_observed_values"]

        # assert context.shape[-1] == self.model.context_length
        # assert target.shape[-1] == self.model.prediction_length

        distr_args, loc, scale = self.model(**select(self.model.input_shapes(), batch))
        distr = self.model.distr_output.distribution(distr_args, loc, scale)

        return (self.loss(distr, target) * observed_target).sum() / torch.maximum(  # type: ignore
            torch.tensor(1.0), observed_target.sum()
        )

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        train_loss = self._compute_loss(batch)
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss = self._compute_loss(batch)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self) -> Any:
        """
        Returns the optimizer to use.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        optimizer_config: Dict[str, Any] = {
            "optimizer": optimizer,
        }

        if self.lr != 0.0:
            optimizer_config["lr_scheduler"] = {
                "scheduler": OneCycleLR(
                    optimizer=optimizer,
                    max_lr=self.lr,
                    epochs=self.epochs,
                    steps_per_epoch=self.steps_per_epoch,
                ),
            }

        return optimizer_config
