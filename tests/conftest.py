# content of conftest.py


from typing import Any, Dict, List

from tpk.utils.dataset import download_datasets


def pytest_configure(config: Dict[str, Any]) -> None:
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """
    download_datasets()


def pytest_collection_modifyitems(items: List[Any]) -> None:
    for item in items:
        item.add_marker("all")
