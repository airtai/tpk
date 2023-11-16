from pathlib import Path
from typing import Dict

import pandas as pd


def _get_m5_dataset_raw(m5_dataset_path: Path) -> Dict[str, pd.DataFrame]:
    dataset = {
        "calendar": pd.read_csv(m5_dataset_path / "calendar.csv"),
        "sales_train_validation": pd.read_csv(
            m5_dataset_path / "sales_train_validation.csv"
        ),
        "sales_train_evaluation": pd.read_csv(
            m5_dataset_path / "sales_train_evaluation.csv"
        ),
        "sample_submission": pd.read_csv(m5_dataset_path / "sample_submission.csv"),
        "sell_prices": pd.read_csv(m5_dataset_path / "sell_prices.csv"),
    }
    return dataset


def _stack_m5_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
    df = df.stack()
    df = df.reset_index(name="n_sold")
    df = df.rename(columns={"level_6": "day"})
    return df


def _combine_m5_sales_and_calendar(
    df_sales: pd.DataFrame, df_calendar: pd.DataFrame
) -> pd.DataFrame:
    return pd.merge(df_sales, df_calendar, left_on="day", right_on="d")


def _combine_m5_sales_and_sell_prices(
    df_sales: pd.DataFrame, df_prices: pd.DataFrame
) -> pd.DataFrame:
    df = pd.merge(
        df_sales, df_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )
    df["sell_price_known"] = df["sell_price"].notna().astype(int)

    mean = df.sell_price.mean(skipna=True)
    df["sell_price"] = df["sell_price"].fillna(mean)

    return df


def _combine_dataset(
    df_sales: pd.DataFrame, df_calendar: pd.DataFrame, df_prices: pd.DataFrame
) -> pd.DataFrame:
    sales_df_days = _combine_m5_sales_and_calendar(df_sales, df_calendar)
    complete_df = _combine_m5_sales_and_sell_prices(sales_df_days, df_prices)
    return complete_df


def get_m5_dataset(m5_dataset_path: Path) -> pd.DataFrame:
    dataset = _get_m5_dataset_raw(m5_dataset_path)

    return _combine_dataset(
        dataset["sales_train_validation"], dataset["calendar"], dataset["sell_prices"]
    ), _combine_dataset(
        dataset["sales_train_evaluation"], dataset["calendar"], dataset["sell_prices"]
    )
