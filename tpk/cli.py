__all__ = ["app", "hello", "run_study"]


from pathlib import Path
from typing import Literal, Type, Union

import typer

from .torch import TPKModel, TSMixerModel

app = typer.Typer()


def get_model_cls(
    model_str: Literal["tpk", "tsmixer"]
) -> Type[Union[TPKModel, TSMixerModel]]:
    if model_str == "tpk":
        return TPKModel  # type: ignore[no-any-return]
    elif model_str == "tsmixer":
        return TSMixerModel  # type: ignore[no-any-return]
    else:
        raise ValueError(f"Unknown model: {model_str}")


@app.command()
def train_model(
    model_cls: Literal["tpk", "tsmixer"],
    data_path: str = "data/m5",
    batch_size: int = 64,
    epochs: int = 300,
    context_length: int = 30,
    n_block: int = 2,
    hidden_size: int = 256,
    weight_decay: float = 0.0001,
    dropout_rate: float = 0.0001,
    disable_future_feat: bool = False,
    use_static_feat: bool = True,
) -> None:
    from tpk.hypervalidation import train_model as concrete_train_model

    validation_wrmsse = concrete_train_model(
        model_cls=get_model_cls(model_cls),
        data_path=data_path,
        batch_size=batch_size,
        epochs=epochs,
        context_length=context_length,
        n_block=n_block,
        hidden_size=hidden_size,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        disable_future_feature=disable_future_feat,
        use_static_feat=use_static_feat,
    )

    typer.echo(validation_wrmsse)


@app.command()
def run_study(
    model_cls: Literal["tpk", "tsmixer"],
    study_name: str,
    n_trials: int,
    tests_per_trial: int = 5,
    study_journal_path: str = "data/journal",
    data_path: str = "data/m5",
) -> None:
    from tpk.hypervalidation import run_study as concrete_run_study

    concrete_run_study(
        model_cls=model_cls,
        study_journal_path=Path(study_journal_path),
        data_path=Path(data_path),
        study_name=study_name,
        n_trials=n_trials,
        tests_per_trial=tests_per_trial,
    )


@app.command()
def hello() -> None:
    print("Hi from temporal predictions kit")
