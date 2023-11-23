import argparse
import os
import time
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.forecast import QuantileForecast
from gluonts.torch.distributions import NegativeBinomialOutput
from gluonts.torch.model.forecast import DistributionForecast as PTDistributionForecast
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from tqdm import tqdm

from temporal_data_kit.datasets.m5_tsmixer import (
    N_TS,
    PREDICTION_LENGTH,
    TEST_START,
    VAL_START,
    load_datasets,
)
from temporal_data_kit.models.TSMixer.accuracy_evaluator import evaluate_wrmsse
from temporal_data_kit.models.TSMixer.estimator import TSMixerEstimator


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--ckpt_dir", default="./ckpt")
    parser.add_argument("--model", default="tsmixer")
    parser.add_argument("--seq_len", type=int, default=35)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--n_stack", type=int, default=30)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--temporal_hidden_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable_static", action="store_true")
    parser.add_argument("--disable_future", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--result_path", default="result.csv")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_tsmixer_estimator(
    args: Any, cardinality: List[int], ckpt_dir: str, callbacks: Any
) -> TSMixerEstimator:
    estimator = TSMixerEstimator(
        prediction_length=PREDICTION_LENGTH,
        context_length=args.seq_len,
        n_block=args.n_block,
        hidden_size=args.hidden_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        num_feat_dynamic_real=7,
        disable_future_feature=args.disable_future,
        num_feat_static_cat=0 if args.disable_static else 5,
        cardinality=cardinality,
        batch_size=args.batch_size,
        freq="D",
        distr_output=NegativeBinomialOutput(),
        num_batches_per_epoch=1 if args.debug else (N_TS // args.batch_size + 1),
        trainer_kwargs={
            "accelerator": "gpu",
            "devices": 1,
            "max_epochs": 1 if args.debug else 300,
            "callbacks": callbacks,
            # "ckpt_path": ckpt_dir,
        },
    )
    return estimator  # type: ignore


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
    if isinstance(
        forecasts[0], (PTDistributionForecast, QuantileForecast)
    ):  # MXDistributionForecast,
        for i in range(len(forecasts)):
            forecasts_acc[i] = forecasts[i].mean
    else:
        for i in range(len(forecasts)):
            forecasts_acc[i] = np.mean(forecasts[i].samples, axis=0)
    wrmsse = evaluate_wrmsse(data_dir, forecasts_acc, prediction_start, score_only=True)
    return wrmsse


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    exp_id = f"{args.model}_nb{args.n_block}_dp{args.dropout}_hs{args.hidden_size}_ds{int(args.disable_static)}_df{int(args.disable_future)}_s{args.seed}"

    train_ds, val_ds, test_ds, stat_cat_cardinalities = load_datasets(args.data_dir)

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=args.patience)
    ckpt_dir = f"{args.ckpt_dir}/{exp_id}"
    estimator = get_tsmixer_estimator(
        args,
        cardinality=stat_cat_cardinalities,
        ckpt_dir=ckpt_dir,
        callbacks=[early_stop_callback],
    )

    start_training_time = time.time()
    predictor = estimator.train(train_ds, validation_data=val_ds, num_workers=8)
    end_training_time = time.time()
    elasped_training_time = end_training_time - start_training_time
    print(f"Training finished in {elasped_training_time} secconds")

    val_wrmsse = evaluate(args.data_dir, val_ds, predictor, VAL_START, debug=args.debug)
    print(f"val wrmsse: {val_wrmsse}")
    test_wrmsse = evaluate(
        args.data_dir, test_ds, predictor, TEST_START, debug=args.debug
    )
    print(f"test wrmsse: {test_wrmsse}")

    if "tsmixer" in args.model:
        args.model = f"tsmixer_ds{(args.disable_static)}_df{(args.disable_future)}"

    data = [
        {
            "model": args.model,
            "seq_len": args.seq_len,
            "val_wrmsse": val_wrmsse,
            "test_wrmsse": test_wrmsse,
            "training_time": elasped_training_time,
            "n_block": args.n_block,
            "temporal_hidden_size": args.temporal_hidden_size,
            "hidden_size": args.hidden_size,
            "lr": args.lr,
            "dropout": args.dropout,
            "n_stack": args.n_stack,
            "n_head": args.n_head,
            "seed": args.seed,
        }
    ]

    df = pd.DataFrame.from_records(data)
    if os.path.exists(args.result_path):
        df.to_csv(args.result_path, mode="a", index=False, header=False)
    else:
        df.to_csv(args.result_path, mode="w", index=False, header=True)


if __name__ == "__main__":
    main()
