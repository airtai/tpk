import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from temporal_data_kit.hypervalidation.hyperparameter_search import (
    run_model_cmd_parallel,
    run_study,
)


@pytest.mark.asyncio
async def test_num_workers() -> None:
    results = await run_model_cmd_parallel("echo 1", num_executions=3)

    assert results == [1.0, 1.0, 1.0]


@pytest.mark.asyncio
async def test_malformed_return_value() -> None:
    with unittest.TestCase().assertRaises(ValueError) as exc:
        results = await run_model_cmd_parallel("echo hi", num_executions=3)


@pytest.mark.slow
def test_run_study() -> None:
    with TemporaryDirectory() as dir:
        study_journal_path = Path(dir)

        run_study(
            study_journal_path=study_journal_path,
            data_path=Path("data/m5"),
            study_name="test_study",
            n_trials=1,
            tests_per_trial=1,
        )

        assert (study_journal_path / "journal.log").exists()
