from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

PREDICTION_LENGTH = 28
N_TS = 30490
VAL_START = 1886  # 1969 - 3 * 28 + 1
TEST_START = 1914  # 1969 - 2 * 28 + 1


def convert_price_file(m5_input_path: str) -> None:
    # load data
    calendar = pd.read_csv(f"{m5_input_path}/calendar.csv")
    sales_train_evaluation = pd.read_csv(f"{m5_input_path}/sales_train_evaluation.csv")
    sell_prices = pd.read_csv(f"{m5_input_path}/sell_prices.csv")

    # assign price for all days
    week_and_day = calendar[["wm_yr_wk", "d"]]

    price_all_days_items = pd.merge(
        week_and_day, sell_prices, on=["wm_yr_wk"], how="left"
    )  # join on week number
    price_all_days_items = price_all_days_items.drop(["wm_yr_wk"], axis=1)

    # convert days to column
    price_all_items = price_all_days_items.pivot_table(
        values="sell_price", index=["store_id", "item_id"], columns="d"
    )
    price_all_items.reset_index(drop=False, inplace=True)

    # reorder column
    price_all_items = price_all_items.reindex(
        ["store_id", "item_id"] + ["d_%d" % x for x in range(1, 1969 + 1)], axis=1
    )

    sales_keys = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales_keys_pd = sales_train_evaluation[sales_keys]

    # join with sales data
    price_converted = pd.merge(
        sales_keys_pd, price_all_items, on=["store_id", "item_id"], how="left"
    )

    # save file
    price_converted.to_csv(
        f"{m5_input_path}/converted_price_evaluation.csv", index=False
    )


def load_datasets(
    data_dir: str,
) -> Tuple[ListDataset, ListDataset, ListDataset, List[int]]:
    calendar = pd.read_csv(f"{data_dir}/calendar.csv")
    sales_train_evaluation = pd.read_csv(f"{data_dir}/sales_train_evaluation.csv")

    cal_features = calendar.drop(
        [
            "date",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_name_2",
            "d",
        ],
        axis=1,
    )
    cal_features["event_type_1"] = cal_features["event_type_1"].apply(
        lambda x: 0 if str(x) == "nan" else 1
    )
    cal_features["event_type_2"] = cal_features["event_type_2"].apply(
        lambda x: 0 if str(x) == "nan" else 1
    )

    event_features = cal_features.values.T
    event_features_expand = np.tile(event_features, (len(sales_train_evaluation), 1, 1))

    state_ids = sales_train_evaluation["state_id"].astype("category").cat.codes.values
    state_ids_un, state_ids_counts = np.unique(state_ids, return_counts=True)

    store_ids = sales_train_evaluation["store_id"].astype("category").cat.codes.values
    store_ids_un, store_ids_counts = np.unique(store_ids, return_counts=True)

    cat_ids = sales_train_evaluation["cat_id"].astype("category").cat.codes.values
    cat_ids_un, cat_ids_counts = np.unique(cat_ids, return_counts=True)

    dept_ids = sales_train_evaluation["dept_id"].astype("category").cat.codes.values
    dept_ids_un, dept_ids_counts = np.unique(dept_ids, return_counts=True)

    item_ids = sales_train_evaluation["item_id"].astype("category").cat.codes.values
    item_ids_un, item_ids_counts = np.unique(item_ids, return_counts=True)

    stat_cat_list = [item_ids, dept_ids, cat_ids, store_ids, state_ids]
    stat_cat = np.concatenate(stat_cat_list)
    stat_cat = stat_cat.reshape(len(stat_cat_list), len(item_ids)).T  # type: ignore
    stat_cat_cardinalities = [
        len(item_ids_un),
        len(dept_ids_un),
        len(cat_ids_un),
        len(store_ids_un),
        len(state_ids_un),
    ]  # type: ignore

    train_df = sales_train_evaluation.drop(
        ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], axis=1
    )
    train_target_values = train_df.values

    test_target_values = train_target_values.copy()
    train_target_values = [ts[: -(2 * PREDICTION_LENGTH)] for ts in train_df.values]
    val_target_values = [ts[:-PREDICTION_LENGTH] for ts in train_df.values]

    # snap features
    # snap_features = calendar[['snap_CA', 'snap_TX', 'snap_WI']]
    # snap_features = snap_features.values.T
    # snap_features_expand = np.array([snap_features] * len(sales_train_evaluation))    # 30490 * 3 * T

    # sell_prices
    converted_price_file = Path(f"{data_dir}/converted_price_evaluation.csv")
    if not converted_price_file.exists():
        convert_price_file(data_dir)
    converted_price = pd.read_csv(converted_price_file)

    price_feature = converted_price.drop(
        ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], axis=1
    ).values

    # normalized sell prices
    normalized_price_file = Path(f"{data_dir}/normalized_price_evaluation.npz")
    if not normalized_price_file.exists():
        # normalized sell prices per each item
        price_mean_per_item = np.nanmean(price_feature, axis=1, keepdims=True)
        price_std_per_item = np.nanstd(price_feature, axis=1, keepdims=True)
        normalized_price_per_item = (price_feature - price_mean_per_item) / (
            price_std_per_item + 1e-6
        )

        # normalized sell prices per day within the same dept
        dept_groups = converted_price.drop(
            ["id", "item_id", "cat_id", "store_id", "state_id"], axis=1
        ).groupby("dept_id")
        price_mean_per_dept = dept_groups.transform(np.nanmean)
        price_std_per_dept = dept_groups.transform(np.nanstd)
        normalized_price_per_group_pd = (
            converted_price[price_mean_per_dept.columns] - price_mean_per_dept
        ) / (price_std_per_dept + 1e-6)

        normalized_price_per_group = normalized_price_per_group_pd.values
        np.savez(
            normalized_price_file,
            per_item=normalized_price_per_item,
            per_group=normalized_price_per_group,
        )
    else:
        normalized_price = np.load(normalized_price_file)
        normalized_price_per_item = normalized_price["per_item"]
        normalized_price_per_group = normalized_price["per_group"]

    price_feature = np.nan_to_num(price_feature)
    normalized_price_per_item = np.nan_to_num(normalized_price_per_item)
    normalized_price_per_group = np.nan_to_num(normalized_price_per_group)

    all_price_features = np.stack(
        [normalized_price_per_item, normalized_price_per_group], axis=1
    )  # 30490 * 2 * T
    # dynamic_real = np.concatenate([snap_features_expand, all_price_features, event_features_expand], axis=1)    # 30490 * 6 * T
    dynamic_real = np.concatenate(
        [all_price_features, event_features_expand], axis=1
    )  # 30490 * 6 * T

    train_dynamic_real = dynamic_real[..., : VAL_START - 1]
    val_dynamic_real = dynamic_real[..., : TEST_START - 1]
    test_dynamic_real = dynamic_real[..., :-PREDICTION_LENGTH]

    m5_dates = [pd.Timestamp("2011-01-29") for _ in range(len(sales_train_evaluation))]

    train_ds = ListDataset(
        [
            {
                FieldName.TARGET: target,
                FieldName.START: start,
                FieldName.FEAT_DYNAMIC_REAL: fdr,
                FieldName.FEAT_STATIC_CAT: fsc,
            }
            for (target, start, fdr, fsc) in zip(
                train_target_values, m5_dates, train_dynamic_real, stat_cat
            )
        ],
        freq="D",
    )

    val_ds = ListDataset(
        [
            {
                FieldName.TARGET: target,
                FieldName.START: start,
                FieldName.FEAT_DYNAMIC_REAL: fdr,
                FieldName.FEAT_STATIC_CAT: fsc,
            }
            for (target, start, fdr, fsc) in zip(
                val_target_values, m5_dates, val_dynamic_real, stat_cat
            )
        ],
        freq="D",
    )

    test_ds = ListDataset(
        [
            {
                FieldName.TARGET: target,
                FieldName.START: start,
                FieldName.FEAT_DYNAMIC_REAL: fdr,
                FieldName.FEAT_STATIC_CAT: fsc,
            }
            for (target, start, fdr, fsc) in zip(
                test_target_values, m5_dates, test_dynamic_real, stat_cat
            )
        ],
        freq="D",
    )

    return train_ds, val_ds, test_ds, stat_cat_cardinalities
