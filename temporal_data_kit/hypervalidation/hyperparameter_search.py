import asyncio
import shlex
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

import asyncer
import numpy as np
import optuna  # type: ignore[import]
import typer


async def run_model_cmd_parallel(model_cmd: str, num_executions: int) -> List[float]:
    async with asyncer.create_task_group() as tg:
        tasks = []
        for _ in range(num_executions):
            tasks.append(
                tg.soonify(asyncio.create_subprocess_exec)(
                    *shlex.split(model_cmd),
                    stdout=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.PIPE,
                )
            )
            await asyncio.sleep(0.001)

    procs = [task.value for task in tasks]

    async def log_output(
        output: Optional[asyncio.StreamReader],
        pid: int,
    ) -> float:
        if output is None:
            raise RuntimeError("Expected StreamReader, got None. Is stdout piped?")
        last_out = ""
        while not output.at_eof():
            outs = await output.readline()
            if outs != b"":
                typer.echo(f"[{pid:03d}]: " + outs.decode("utf-8"), nl=False)
                last_out = outs.decode("utf-8").strip()
        return float(last_out)

    async with asyncer.create_task_group() as tg:
        soon_values = [tg.soonify(log_output)(proc.stdout, proc.pid) for proc in procs]

    values = [soon_value.value for soon_value in soon_values]

    return values


def objective(
    data_path: str,
    tests_per_trial: int,
) -> Callable[[optuna.Trial], Union[float, Sequence[float]]]:
    def _inner(
        trial: Any,
        data_path: str = data_path,
        tests_per_trial: int = tests_per_trial,
        batch_size: int = 64,
        epochs: int = 1,
        patience: int = 30,
    ) -> float:
        trial_values = {
            "data_path": data_path,
            "context_length": trial.suggest_categorical("context_length", [20, 35, 50]),
            "n_block": trial.suggest_int("n_block", 1, 5),
            "hidden_size": trial.suggest_categorical(
                "hidden_size", [64, 128, 256, 512]
            ),
            "lr": trial.suggest_float("learning_rate", 0.0001, 0.5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0001, 0.5, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0001, 0.5, log=True),
            "disable_future_feat": trial.suggest_categorical(
                "disable_future", [True, False]
            ),
            "use_static_feat": trial.suggest_categorical(
                "use_static_feat", [True, False]
            ),
            "patience": patience,
            "batch_size": batch_size,
            "epochs": epochs,
        }

        values = asyncio.run(
            run_model_cmd_parallel(
                model_cmd="temporal_data_kit train-model --epochs 1",
                num_executions=tests_per_trial,
            )
        )

        return float(np.mean(values))

    return _inner


def run_study(
    study_journal_path: Path,
    study_name: str,
    data_path: Path,
    n_trials: int,
    tests_per_trial: int,
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
    study.optimize(
        objective(str(data_path), tests_per_trial),
        n_trials=n_trials,
        catch=(ValueError,),
    )  # Model diverging with ValueError, catch and go to next trial
