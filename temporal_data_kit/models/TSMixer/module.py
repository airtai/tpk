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

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonts.core.component import validated
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.scaler import MeanScaler, NOPScaler


class TemporalLinear(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        activation: Optional[str] = None,
        dropout: float = 0,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=input_len, out_features=output_len)
        self.activation = None if activation is None else getattr(F, activation)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x if self.activation is None else self.activation(x)
        x = self.dropout(x)
        return x


class TemporalResBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_size: int,
        activation: Optional[str] = None,
        dropout: float = 0,
    ):
        super().__init__()
        self.temporal_linear = TemporalLinear(input_len, input_len, activation, dropout)
        self.norm = nn.LayerNorm(normalized_shape=[input_len, input_size])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.temporal_linear(x)
        return self.norm(res + x)  # type: ignore


class FeaturalResBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: str = "relu",
        dropout: float = 0,
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.res_linear = None
        if input_size != output_size:
            self.res_linear = nn.Linear(
                in_features=input_size, out_features=output_size
            )
        self.activation = getattr(F, activation)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(normalized_shape=[input_len, output_size])

    def forward(self, x: torch.Tensor) -> Any:
        res = x if self.res_linear is None else self.res_linear(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.norm(res + x)


class ConditionalFeaturalResBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        static_size: int,
        activation: str = "relu",
        dropout: float = 0,
    ):
        super().__init__()
        self.input_len = input_len
        self.static_block = FeaturalResBlock(
            1, static_size, hidden_size, hidden_size, activation, dropout
        )
        self.block = FeaturalResBlock(
            input_len,
            input_size + hidden_size,
            hidden_size,
            output_size,
            activation,
            dropout,
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        static = self.static_block(static.unsqueeze(1))
        static = torch.repeat_interleave(static, self.input_len, dim=1)
        x = torch.concat([x, static], dim=2)
        x = self.block(x)
        return x


class MixerBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        self.temporal_res_block = TemporalResBlock(
            input_len, input_size, activation, dropout
        )
        self.ffwd_res_block = FeaturalResBlock(
            input_len,
            input_size,
            hidden_size,
            output_size,
            activation,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_res_block(x)
        x = self.ffwd_res_block(x)
        return x


class ConditionalMixerBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        static_size: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        self.temporal_res_block = TemporalResBlock(
            input_len, input_size, activation, dropout
        )
        self.ffwd_res_block = ConditionalFeaturalResBlock(
            input_len,
            input_size,
            hidden_size,
            output_size,
            static_size,
            activation,
            dropout,
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        x = self.temporal_res_block(x)
        x = self.ffwd_res_block(x, static)
        return x


class TSMixerEncoder(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        past_feat_size: int,
        future_feat_size: int,
        static_feat_size: int,
        hidden_size: int,
        activation: str,
        dropout: float,
        n_block: int = 1,
    ):
        super().__init__()
        self.past_temporal_linear = TemporalLinear(input_len, output_len)
        self.past_featural_block = ConditionalFeaturalResBlock(
            input_len=output_len,
            input_size=past_feat_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            static_size=static_feat_size,
            activation=activation,
            dropout=dropout,
        )
        self.future_featural_block = ConditionalFeaturalResBlock(
            input_len=output_len,
            input_size=future_feat_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            static_size=static_feat_size,
            activation=activation,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                ConditionalMixerBlock(
                    input_len=output_len,
                    input_size=(2 * hidden_size) if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    static_size=static_feat_size,
                    activation=activation,
                    dropout=dropout,
                )
                for i in range(n_block)
            ]
        )

    def forward(
        self,
        past_feature: torch.Tensor,
        future_feature: torch.Tensor,
        static_feature: torch.Tensor,
    ) -> torch.Tensor:
        past_feature = self.past_temporal_linear(past_feature)
        past_feature = self.past_featural_block(past_feature, static_feature)
        future_feature = self.future_featural_block(future_feature, static_feature)
        x = torch.cat([past_feature, future_feature], dim=2)
        for block in self.blocks:
            x = block(x, static_feature)
        return x


class TSMixerModel(nn.Module):
    """
    Module implementing the TSMixer model, see [SFG17]_.

    *Note:* the code of this model is unrelated to the implementation behind
    `SageMaker's TSMixer Forecasting Algorithm
    <https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html>`_.

    Parameters
    ----------
    freq
        String indicating the sampling frequency of the data to be processed.
    context_length
        Length of the RNN unrolling prior to the forecast date.
    prediction_length
        Number of time points to predict.
    num_feat_dynamic_real
        Number of dynamic real features that will be provided to ``forward``.
    num_feat_static_real
        Number of static real features that will be provided to ``forward``.
    num_feat_static_cat
        Number of static categorical features that will be provided to
        ``forward``.
    cardinality
        List of cardinalities, one for each static categorical feature.
    embedding_dimension
        Dimension of the embedding space, one for each static categorical
        feature.
    n_block
        Number of layers in the RNN.
    hidden_size
        Size of the hidden layers in the RNN.
    dropout_rate
        Dropout rate to be applied at training time.
    distr_output
        Type of distribution to be output by the model at each time step
    scaling
        Whether to apply mean scaling to the observations (target).
    """

    @validated()  # type: ignore
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_future_feat: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        distr_output: DistributionOutput,
        embedding_dimension: Optional[List[int]] = None,
        n_block: int = 2,
        hidden_size: int = 128,
        dropout_rate: float = 0.1,
        scaling: bool = True,
    ) -> None:
        super().__init__()

        # assert distr_output.event_shape == ()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_future_feat = num_future_feat
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(32, (cat + 1) // 2) for cat in cardinality]
        )
        self.past_length = self.context_length
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )

        if scaling:
            self.scaler = MeanScaler(dim=-1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=-1, keepdim=True)

        self.distr_output = StudentTOutput() if distr_output is None else distr_output
        self.args_proj = distr_output.get_args_proj(hidden_size)

        self.tsmixer_encoder = TSMixerEncoder(
            input_len=context_length,
            output_len=prediction_length,
            past_feat_size=self.num_feat_dynamic_real + 1,  # target
            future_feat_size=max(1, self.num_future_feat),
            static_feat_size=(sum(self.embedding_dimension) + num_feat_static_real + 1),
            activation="relu",
            dropout=dropout_rate,
            n_block=n_block,
            hidden_size=hidden_size,
        )

    @property
    def _past_length(self) -> int:
        return self.context_length

    def input_shapes(self, batch_size: int = 1) -> Dict[str, Tuple[int, ...]]:
        return {
            "feat_static_cat": (batch_size, self.num_feat_static_cat),
            "feat_static_real": (batch_size, self.num_feat_static_real),
            "past_time_feat": (
                batch_size,
                self._past_length,
                self.num_feat_dynamic_real,
            ),
            "past_target": (batch_size, self._past_length),
            "past_observed_values": (batch_size, self._past_length),
            "future_time_feat": (
                batch_size,
                self.prediction_length,
                self.num_feat_dynamic_real,
            ),
        }

    def input_types(self) -> Dict[str, torch.dtype]:
        return {
            "feat_static_cat": torch.long,
            "feat_static_real": torch.float,
            "past_time_feat": torch.float,
            "past_target": torch.float,
            "past_observed_values": torch.float,
            "future_time_feat": torch.float,
        }

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> Tuple[Any, torch.Tensor, Any]:
        """
        Invokes the model on input data, and produce outputs future samples.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            (Optional) tensor of dynamic real features in the past,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        """
        if self.num_future_feat == 0:
            future_time_feat = torch.zeros_like(future_time_feat)[:, :, [0]]
        else:
            future_time_feat = future_time_feat[:, :, : self.num_future_feat]
        _, _, scale = self.scaler(past_target, past_observed_values)

        scaled_past_target = past_target / scale

        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, scale.log()),
            dim=-1,
        )
        past_feature = torch.cat(
            (scaled_past_target.unsqueeze(-1), past_time_feat), dim=-1
        )
        output = self.tsmixer_encoder(past_feature, future_time_feat, static_feat)
        distr_args = self.args_proj(output)
        return distr_args, torch.zeros_like(scale), scale
