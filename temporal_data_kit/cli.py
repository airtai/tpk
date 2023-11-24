__all__ = ["app", "run_study", "hello"]


from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def run_study(
    study_name: str,
    n_trials: int,
    study_journal_path: str = "data/journal",
    data_path: str = "data/m5",
) -> None:
    from temporal_data_kit.hypervalidation import run_study as concrete_run_study

    concrete_run_study(
        study_journal_path=Path(study_journal_path),
        data_path=Path(data_path),
        study_name=study_name,
        n_trials=n_trials,
    )


@app.command()
def hello() -> None:
    print("Hi from temporal data kit")
