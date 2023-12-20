__all__ = ["app", "run_study"]


from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Type, Union

import typer
from typing_extensions import Annotated

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
    model_cls: Annotated[
        str,
        typer.Option(help="Class of the model to be trained, can be: tsmixer or tpk"),
    ] = "tsmixer",
    data_path: Annotated[str, typer.Option(help="Path to the dataset")] = "data/m5",
    batch_size: Annotated[int, typer.Option(help="Batch size for model training")] = 64,
    epochs: Annotated[
        int, typer.Option(help="Number of epochs to run the model training")
    ] = 300,
    context_length: Annotated[
        int, typer.Option(help="Context length of the model")
    ] = 30,
    n_block: Annotated[int, typer.Option(help="Number of model hidden blocks")] = 2,
    hidden_size: Annotated[int, typer.Option(help="Size of hidden layers")] = 256,
    weight_decay: Annotated[float, typer.Option(help="Model weight decay")] = 0.0001,
    lr: Annotated[float, typer.Option(help="Model learning rate")] = 0.0001,
    dropout_rate: Annotated[float, typer.Option(help="Model dropout rate")] = 0.0001,
    disable_future_feat: Annotated[
        bool, typer.Option(help="Disable future features")
    ] = False,
    use_static_feat: Annotated[bool, typer.Option(help="Use static features")] = True,
) -> None:
    from tpk.hypervalidation import train_model as concrete_train_model

    validation_wrmsse = concrete_train_model(
        model_cls=get_model_cls(model_cls),  # type: ignore
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
        lr=lr,
    )

    typer.echo(validation_wrmsse)


@app.command()
def run_study(
    model_cls: Annotated[
        str,
        typer.Option(help="Class of the model to be trained, can be: tsmixer or tpk"),
    ] = "tsmixer",
    study_name: Annotated[Optional[str], typer.Option(help="Name of the study")] = None,
    n_trials: Annotated[
        int, typer.Option(help="Number of trial to run in the study")
    ] = 100,
    tests_per_trial: Annotated[
        int, typer.Option(help="Number of test models to run per trial")
    ] = 5,
    study_journal_path: Annotated[
        str, typer.Option(help="Path to study journal")
    ] = "data/journal",
    data_path: Annotated[str, typer.Option(help="Path to the dataset")] = "data/m5",
) -> None:
    from tpk.hypervalidation import run_study as concrete_run_study

    if study_name is None:
        study_name = f"{model_cls}_{datetime.now().isoformat()}_study"

    concrete_run_study(
        model_cls=model_cls,  # type: ignore
        study_journal_path=Path(study_journal_path),
        data_path=Path(data_path),
        study_name=study_name,
        n_trials=n_trials,
        tests_per_trial=tests_per_trial,
    )


@app.command()
def find_lr(
    model_cls: Annotated[
        str,
        typer.Option(help="Class of the model to be trained, can be: tsmixer or tpk"),
    ] = "tsmixer",
    data_path: Annotated[str, typer.Option(help="Path to the dataset")] = "data/m5",
    batch_size: Annotated[int, typer.Option(help="Batch size for model training")] = 64,
    epochs: Annotated[
        int, typer.Option(help="Number of epochs to run the model training")
    ] = 300,
    context_length: Annotated[
        int, typer.Option(help="Context length of the model")
    ] = 30,
    n_block: Annotated[int, typer.Option(help="Number of model hidden blocks")] = 2,
    hidden_size: Annotated[int, typer.Option(help="Size of hidden layers")] = 256,
    weight_decay: Annotated[float, typer.Option(help="Model weight decay")] = 0.0001,
    dropout_rate: Annotated[float, typer.Option(help="Model dropout rate")] = 0.0001,
    disable_future_feat: Annotated[
        bool, typer.Option(help="Disable future features")
    ] = False,
    use_static_feat: Annotated[bool, typer.Option(help="Use static features")] = True,
) -> None:
    from tpk.hypervalidation import find_lr as concrete_find_lr

    lr = concrete_find_lr(
        model_cls=get_model_cls(model_cls),  # type: ignore
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
        lr=0.0,
    )

    typer.echo(lr)
