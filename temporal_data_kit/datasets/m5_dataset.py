from pathlib import Path
from typing import Tuple

import pandas as pd
from torch.utils.data import Dataset


def _stack_m5_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
    df = df.stack()
    df = df.reset_index(name="n_sold")
    df = df.rename(columns={"level_6": "day"})
    return df


def _combine_m5_sales_and_calendar(
    df_sales: pd.DataFrame, df_calendar: pd.DataFrame
) -> pd.DataFrame:
    return pd.merge(
        df_sales,
        df_calendar.drop(
            columns=[
                "weekday",
                "wday",
                "month",
                "year",
                "snap_CA",
                "snap_TX",
                "snap_WI",
            ]
        ),
        left_on="day",
        right_on="d",
    ).drop(columns=["day", "d"])


def _combine_m5_sales_and_sell_prices(
    df_sales: pd.DataFrame, df_prices: pd.DataFrame
) -> pd.DataFrame:
    df = pd.merge(
        df_sales, df_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )
    df["sell_price_known"] = df["sell_price"].notna().astype(int)

    mean = df.sell_price.mean(skipna=True)
    df["sell_price"] = df["sell_price"].fillna(mean)

    return df.drop(columns=["wm_yr_wk"])


def _combine_dataset(
    df_sales: pd.DataFrame, df_calendar: pd.DataFrame, df_prices: pd.DataFrame
) -> pd.DataFrame:
    sales_df_stacked = _stack_m5_days(df_sales)
    sales_df_days = _combine_m5_sales_and_calendar(sales_df_stacked, df_calendar)
    complete_df = _combine_m5_sales_and_sell_prices(sales_df_days, df_prices)
    return complete_df


def get_m5_dataset(m5_dataset_path: Path, test: bool = False) -> pd.DataFrame:
    df_calendar = pd.read_csv(m5_dataset_path / "calendar.csv")
    df_sales = pd.read_csv(
        m5_dataset_path
        / (("sales_train_" + "validation" if not test else "evaluation") + ".csv")
    )
    df_prices = pd.read_csv(m5_dataset_path / "sell_prices.csv")

    return _combine_dataset(
        df_sales=df_sales, df_calendar=df_calendar, df_prices=df_prices
    )


class M5(Dataset):  # type: ignore
    def __init__(  # type: ignore
        self, m5_dataset_path: Path = Path("./data/m5/"), test: bool = False, **kwargs
    ):
        self.dataset = get_m5_dataset(
            m5_dataset_path=m5_dataset_path, test=test, **kwargs
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[pd.Series, int]:
        sample = self.dataset.iloc[idx]
        return sample.drop("n_sold"), sample["n_sold"]
