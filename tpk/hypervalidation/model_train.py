from typing import Any, Type, Union

import numpy as np
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.forecast import QuantileForecast
from gluonts.torch.distributions import NegativeBinomialOutput
from gluonts.torch.model.forecast import DistributionForecast as PTDistributionForecast
from tqdm import tqdm

from tpk.testing.datasets.m5 import (
    N_TS,
    PREDICTION_LENGTH,
    VAL_START,
    evaluate_wrmsse,
    load_datasets,
)
from tpk.torch import MyEstimator, TPKModel, TSMixerModel


def evaluate(
    data_dir: str,
    dataset: Any,
    predictor: Any,
    prediction_start: int,
) -> Any:
    forecast_it, _ = make_evaluation_predictions(
        dataset=dataset, predictor=predictor, num_samples=100
    )

    forecasts = list(tqdm(forecast_it, total=len(dataset)))

    forecasts_acc = np.zeros((len(forecasts), PREDICTION_LENGTH))
    if isinstance(forecasts[0], (PTDistributionForecast, QuantileForecast)):
        for i in range(len(forecasts)):
            forecasts_acc[i] = forecasts[i].mean
    else:
        for i in range(len(forecasts)):
            forecasts_acc[i] = np.mean(forecasts[i].samples, axis=0)
    wrmsse = evaluate_wrmsse(data_dir, forecasts_acc, prediction_start, score_only=True)
    return wrmsse


def train_model(
    *,
    model_cls: Type[Union[TPKModel, TSMixerModel]],
    data_path: str,
    batch_size: int,
    epochs: int,
    context_length: int,
    n_block: int,
    hidden_size: int,
    weight_decay: float,
    dropout_rate: float,
    disable_future_feature: bool,
    use_static_feat: bool,
    lr: float,
    use_one_cycle: bool,
) -> float:
    train_ds, val_ds, _, stat_cat_cardinalities = load_datasets(data_path)
    estimator = MyEstimator(
        model_cls=model_cls,
        prediction_length=PREDICTION_LENGTH,
        context_length=context_length,
        epochs=epochs,
        n_block=n_block,
        hidden_size=hidden_size,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        num_feat_dynamic_real=7,
        disable_future_feature=disable_future_feature,
        num_feat_static_cat=5 if use_static_feat else 0,
        cardinality=stat_cat_cardinalities,
        batch_size=batch_size,
        lr=lr,
        freq="D",
        distr_output=NegativeBinomialOutput(),
        num_batches_per_epoch=(N_TS // batch_size + 1),
        trainer_kwargs={
            "accelerator": "gpu",
            "devices": 1,
            "max_epochs": epochs,
            "callbacks": [],
        },
        use_one_cycle=use_one_cycle,
    )

    predictor = estimator.train(train_ds, validation_data=val_ds, num_workers=32)

    val_wrmsse = evaluate(data_path, val_ds, predictor, VAL_START)
    return val_wrmsse  # type: ignore


def find_lr(
    *,
    model_cls: Type[Union[TPKModel, TSMixerModel]],
    data_path: str,
    batch_size: int,
    epochs: int,
    context_length: int,
    n_block: int,
    hidden_size: int,
    weight_decay: float,
    dropout_rate: float,
    disable_future_feature: bool,
    use_static_feat: bool,
    lr: float,
) -> float:
    train_ds, _, _, stat_cat_cardinalities = load_datasets(data_path)
    estimator = MyEstimator(
        model_cls=model_cls,
        prediction_length=PREDICTION_LENGTH,
        context_length=context_length,
        epochs=epochs,
        n_block=n_block,
        hidden_size=hidden_size,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        num_feat_dynamic_real=7,
        disable_future_feature=disable_future_feature,
        num_feat_static_cat=5 if use_static_feat else 0,
        cardinality=stat_cat_cardinalities,
        batch_size=batch_size,
        lr=lr,
        freq="D",
        distr_output=NegativeBinomialOutput(),
        num_batches_per_epoch=(N_TS // batch_size + 1),
        trainer_kwargs={
            "accelerator": "gpu",
            "devices": 1,
            "max_epochs": epochs,
            "callbacks": [],
        },
    )

    return estimator.find_lr(train_ds)  # type: ignore
