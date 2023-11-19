from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from temporal_data_kit.datasets.m5_dataset import (
    M5,
    _combine_m5_sales_and_calendar,
    _combine_m5_sales_and_sell_prices,
    _stack_m5_days,
)

sales_train_validation = pd.DataFrame(
    data={
        "id": ["HOBBIES_1_001_CA_1_validation", "HOBBIES_1_002_WI_1_validation"],
        "item_id": ["HOBBIES_1_001", "HOBBIES_1_002"],
        "dept_id": ["HOBBIES_1", "HOBBIES_1"],
        "cat_id": ["HOBBIES", "HOBBIES"],
        "store_id": ["CA_1", "WI_1"],
        "state_id": ["CA", "WI"],
        "d_1": [0, 1],
        "d_2": [1, 2],
    }
)

calendar = pd.DataFrame(
    data={
        "date": ["2011-01-29", "2011-01-30"],
        "wm_yr_wk": [1101, 1102],
        "weekday": ["Saturday", "Sunday"],
        "wday": [1, 1],
        "month": [1, 1],
        "year": [2011, 2011],
        "d": ["d_1", "d_2"],
        "event_name_1": [None, "Fathers Day"],
        "event_type_1": [None, "Holiday"],
        "event_name_2": [None, None],
        "event_type_2": [None, None],
        "snap_CA": [0, 0],
        "snap_TX": [0, 0],
        "snap_WI": [0, 0],
    }
)

sell_prices = pd.DataFrame(
    data={
        "store_id": ["CA_1", "WI_1"],
        "item_id": ["HOBBIES_1_001", "HOBBIES_1_002"],
        "wm_yr_wk": [1101, 1102],
        "sell_price": [9.99, 11.59],
    }
)


def test_stack_sales():
    assert _stack_m5_days(sales_train_validation).equals(
        pd.DataFrame(
            data={
                "id": [
                    "HOBBIES_1_001_CA_1_validation",
                    "HOBBIES_1_001_CA_1_validation",
                    "HOBBIES_1_002_WI_1_validation",
                    "HOBBIES_1_002_WI_1_validation",
                ],
                "item_id": [
                    "HOBBIES_1_001",
                    "HOBBIES_1_001",
                    "HOBBIES_1_002",
                    "HOBBIES_1_002",
                ],
                "dept_id": ["HOBBIES_1"] * 4,
                "cat_id": ["HOBBIES"] * 4,
                "store_id": ["CA_1", "CA_1", "WI_1", "WI_1"],
                "state_id": ["CA", "CA", "WI", "WI"],
                "day": ["d_1", "d_2", "d_1", "d_2"],
                "n_sold": [0, 1, 1, 2],
            }
        )
    )


def test_combine_calendar():
    combined_dates = _combine_m5_sales_and_calendar(
        _stack_m5_days(sales_train_validation), calendar
    )
    assert (
        combined_dates["event_name_1"].fillna("-")
        == ["-", "-", "Fathers Day", "Fathers Day"]
    ).all()
    assert (
        combined_dates["event_type_1"].fillna("-") == ["-", "-", "Holiday", "Holiday"]
    ).all()
    assert (combined_dates["event_name_2"].fillna("-") == ["-"] * 4).all()
    assert (combined_dates["event_type_2"].fillna("-") == ["-"] * 4).all()
    assert (
        combined_dates["wm_yr_wk"] == [1101, 1101, 1102, 1102]
    ).all(), combined_dates["wm_yr_wk"].fillna("-")


def test_combine_prices():
    combined_prices = _combine_m5_sales_and_sell_prices(
        _combine_m5_sales_and_calendar(
            _stack_m5_days(sales_train_validation), calendar
        ),
        sell_prices,
    )
    assert (combined_prices["sell_price"] == [9.99, 10.79, 10.79, 11.59]).all()
    assert (combined_prices["sell_price_known"] == [1, 0, 0, 1]).all()


def test_full_m5():
    with TemporaryDirectory() as dir:
        m5_path = Path(dir)
        sales_train_validation.to_csv(m5_path / "sales_train_validation.csv")
        calendar.to_csv(m5_path / "calendar.csv")
        sell_prices.to_csv(m5_path / "sell_prices.csv")

        dataset = M5(m5_dataset_path=m5_path)

        assert dataset.__len__() == 4
