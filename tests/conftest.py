# content of conftest.py


from temporal_data_kit.utils.dataset import download_datasets


def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """
    download_datasets()


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker("all")
