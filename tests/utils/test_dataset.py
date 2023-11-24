from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from temporal_data_kit.utils.dataset import download_datasets


@pytest.mark.slow
def test_download_datasets() -> None:
    with TemporaryDirectory(prefix="data_") as tmpdir:
        download_datasets(data_root=tmpdir)
        assert (Path(tmpdir) / "m5" / "calendar.csv").exists()
        download_datasets(data_root=tmpdir)
        assert (Path(tmpdir) / "m5" / "calendar.csv").exists()
