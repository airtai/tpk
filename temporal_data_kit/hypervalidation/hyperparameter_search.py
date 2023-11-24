from pathlib import Path
from typing import Any, Callable, List, Sequence, Union

import numpy as np
import optuna  # type: ignore[import]
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.forecast import QuantileForecast
from gluonts.torch.distributions import NegativeBinomialOutput
from gluonts.torch.model.forecast import DistributionForecast as PTDistributionForecast
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from tqdm import tqdm

from temporal_data_kit.model.estimator import TSMixerEstimator
from temporal_data_kit.testing.datasets.m5 import (
    N_TS,
    PREDICTION_LENGTH,
    VAL_START,
    evaluate_wrmsse,
    load_datasets,
)


def evaluate(
    data_dir: str,
    dataset: Any,
    predictor: Any,
    prediction_start: int,
    debug: bool = False,
) -> Any:
    forecast_it, _ = make_evaluation_predictions(
        dataset=dataset, predictor=predictor, num_samples=100
    )

    if debug:
        forecasts = [next(forecast_it)] * len(dataset)
    else:
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


def objective(
    data_path: str,
    batch_size: int = 64,
    epochs: int = 300,
    patience: int = 30,
    debug: bool = False,
) -> Callable[[optuna.Trial], Union[float, Sequence[float]]]:
    train_ds, val_ds, test_ds, stat_cat_cardinalities = load_datasets(data_path)

    def _inner(
        trial: Any,
        train_ds: ListDataset = train_ds,
        val_ds: ListDataset = val_ds,
        test_ds: ListDataset = test_ds,
        stat_cat_cardinalities: List[int] = stat_cat_cardinalities,
        batch_size: int = batch_size,
        epochs: int = epochs,
        patience: int = patience,
        debug: bool = debug,
    ) -> float:
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=patience)
        estimator = TSMixerEstimator(
            prediction_length=PREDICTION_LENGTH,
            context_length=trial.suggest_categorical("context_length", [20, 35, 50]),
            n_block=trial.suggest_int("n_block", 1, 5),
            hidden_size=trial.suggest_categorical("hidden_size", [64, 128, 256, 512]),
            lr=trial.suggest_float("learning_rate", 0.0001, 0.5, log=True),
            weight_decay=trial.suggest_float("weight_decay", 0.0001, 0.5, log=True),
            dropout_rate=trial.suggest_float("dropout_rate", 0.0001, 0.5, log=True),
            num_feat_dynamic_real=7,
            disable_future_feature=trial.suggest_categorical(
                "disable_future", [True, False]
            ),
            num_feat_static_cat=0
            if trial.suggest_categorical("disable_static", [True, False])
            else 5,
            cardinality=stat_cat_cardinalities,
            batch_size=batch_size,
            freq="D",
            distr_output=NegativeBinomialOutput(),
            num_batches_per_epoch=(N_TS // batch_size + 1),
            trainer_kwargs={
                "accelerator": "gpu",
                "devices": 1,
                "max_epochs": epochs,
                "callbacks": [early_stop_callback],
            },
        )

        predictor = estimator.train(train_ds, validation_data=val_ds, num_workers=32)

        val_wrmsse = evaluate(data_path, val_ds, predictor, VAL_START, debug=debug)
        return val_wrmsse  # type: ignore

    return _inner


def run_study(
    study_journal_path: Path, study_name: str, data_path: Path, n_trials: int
) -> None:
    if not study_journal_path.exists():
        study_journal_path.mkdir(exist_ok=True)

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(str(study_journal_path / "journal.log")),
    )

    study = optuna.create_study(
        storage=storage,
        directions=["minimize"],
        study_name=study_name,
    )
    study.optimize(objective(str(data_path)), n_trials=n_trials)
