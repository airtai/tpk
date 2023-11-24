import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm import tqdm

__all__ = ["download_datasets"]

data_sources = {
    "m5": [
        f"https://zenodo.org/records/10203108/files/{name}?download=1"
        for name in [
            "calendar.csv",
            "sales_train_evaluation.csv",
            "sales_train_validation.csv",
            "sample_submission.csv",
            "sell_prices.csv",
        ]
    ],
}


class DownloadProgressBar(tqdm):  # type: ignore[type-arg]
    def update_to(
        self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None
    ) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path) -> None:
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=str(output_path).split("/")[-1]
    ) as t:
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        urllib.request.urlretrieve(
            url,
            filename=output_path,
            reporthook=t.update_to,  # nosec: [B310:blacklist]
        )


def download_datasets(
    *,
    data_root: Union[Path, str] = "data",
    data_sources: Dict[str, List[str]] = data_sources,
) -> None:
    for dataset, urls in data_sources.items():
        for url in tqdm(urls):
            filename = Path(data_root) / dataset / url.split("/")[-1].split("?")[0]
            filename.parent.mkdir(parents=True, exist_ok=True)
            if not filename.exists():
                download_url(url, filename)
            else:
                print(f"File {filename} already exists. Skipping download.")
    print("All files downloaded.")
