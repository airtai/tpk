import gc
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

prediction_length = 28


# Memory reduction helper function:
def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:  # columns
        col_type = df[col].dtypes
        if col_type in numerics:  # numerics
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


# Fucntion to calculate S weights:
def get_s(roll_mat_csr: csr_matrix, sales: pd.DataFrame, prediction_start: int) -> Any:
    # Rollup sales:
    d_name = ["d_" + str(i) for i in range(1, prediction_start)]
    sales_train_val = roll_mat_csr * sales[d_name].values

    no_sales = np.cumsum(sales_train_val, axis=1) == 0

    # Denominator of RMSSE / RMSSE
    diff = np.diff(sales_train_val, axis=1)
    diff = np.where(no_sales[:, 1:], np.nan, diff)

    weight1 = np.nanmean(diff**2, axis=1)
    weight1[np.isnan(weight1)] = 1e-9

    return weight1


# Functinon to calculate weights:
def get_w(roll_mat_csr: csr_matrix, sale_usd: pd.DataFrame) -> Any:
    """ """
    # Calculate the total sales in USD for each item id:
    total_sales_usd = (
        sale_usd.groupby(["id"], sort=False)["sale_usd"].apply(np.sum).values
    )

    # Roll up total sales by ids to higher levels:
    weight2 = roll_mat_csr * total_sales_usd

    return (
        12 * weight2 / np.sum(weight2)
    )  # weight2/(np.sum(weight2) / 12) : np.sum(weight2)은 모든 합의 12배임


# Function to do quick rollups:
def rollup(roll_mat_csr: csr_matrix, v: Any) -> Any:
    """
    v - np.array of size (30490 rows, n day columns)
    v_rolledup - array of size (n, 42840)
    """
    return roll_mat_csr * v  # (v.T*roll_mat_csr.T).T


# Function to calculate WRMSSE:
def wrmsse(
    error: float, score_only: bool, roll_mat_csr: csr_matrix, s: Any, w: Any, sw: Any
) -> Any:
    """
    preds - Predictions: pd.DataFrame of size (30490 rows, N day columns)
    y_true - True values: pd.DataFrame of size (30490 rows, N day columns)
    sequence_length - np.array of size (42840,)
    sales_weight - sales weights based on last 28 days: np.array (42840,)
    """

    if score_only:
        return (
            np.sum(
                np.sqrt(np.mean(np.square(rollup(roll_mat_csr, error)), axis=1)) * sw
            )
            / 12
        )  # <-used to be mistake here
    else:
        score_matrix = (
            np.square(rollup(roll_mat_csr, error)) * np.square(w)[:, None]
        ) / s[:, None]
        wrmsse_i = np.sqrt(np.mean(score_matrix, axis=1))
        wrmsse_raw = np.sqrt(score_matrix)

        aggregation_count = [1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490]

        idx = 0
        aggregated_wrmsse = np.zeros(12)
        aggregated_wrmsse_per_day = np.zeros([12, prediction_length])
        for i, count in enumerate(aggregation_count):
            endIdx = idx + count
            aggregated_wrmsse[i] = wrmsse_i[idx:endIdx].sum()
            aggregated_wrmsse_per_day[i] = wrmsse_raw[idx:endIdx, :].sum(axis=0)
            idx = endIdx

        # score == aggregated_wrmsse.mean()
        wrmsse = np.sum(wrmsse_i) / 12  # <-used to be mistake here

        return (
            wrmsse,
            aggregated_wrmsse,
            aggregated_wrmsse_per_day,
            score_matrix,
        )


def calculate_and_save_data(
    data_path: str, prediction_start: int
) -> Tuple[Any, Any, Any, Any, Any]:
    # Sales quantities:
    sales = pd.read_csv(data_path + "/sales_train_evaluation.csv")

    # Calendar to get week number to join sell prices:
    calendar = pd.read_csv(data_path + "/calendar.csv")
    calendar = reduce_mem_usage(calendar)

    # Sell prices to calculate sales in USD:
    sell_prices = pd.read_csv(data_path + "/sell_prices.csv")
    sell_prices = reduce_mem_usage(sell_prices)

    # Dataframe with only last 28 days:
    cols = [f"d_{i}" for i in range(prediction_start - 28, prediction_start)]
    data = sales[["id", "store_id", "item_id"] + cols]

    # To long form:
    data = data.melt(
        id_vars=["id", "store_id", "item_id"], var_name="d", value_name="sale"
    )

    # Add week of year column from 'calendar':
    data = pd.merge(data, calendar, how="left", left_on=["d"], right_on=["d"])

    data = data[["id", "store_id", "item_id", "sale", "d", "wm_yr_wk"]]

    # Add weekly price from 'sell_prices':
    data = data.merge(sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    data.drop(columns=["wm_yr_wk"], inplace=True)

    # Calculate daily sales in USD:
    data["sale_usd"] = data["sale"] * data["sell_price"]

    # List of categories combinations for aggregations as defined in docs:
    dummies_list = [
        sales.state_id,
        sales.store_id,
        sales.cat_id,
        sales.dept_id,
        sales.state_id + "_" + sales.cat_id,
        sales.state_id + "_" + sales.dept_id,
        sales.store_id + "_" + sales.cat_id,
        sales.store_id + "_" + sales.dept_id,
        sales.item_id,
        sales.state_id + "_" + sales.item_id,
        sales.id,
    ]

    ## First element Level_0 aggregation 'all_sales':
    dummies_df_list = [
        pd.DataFrame(
            np.ones(sales.shape[0]).astype(np.int8),
            index=sales.index,
            columns=["all"],
        ).T
    ]

    # List of dummy dataframes:
    for _, cats in enumerate(dummies_list):
        cat_dtype = pd.api.types.CategoricalDtype(
            categories=pd.unique(cats.values), ordered=True
        )
        ordered_cat = cats.astype(cat_dtype)
        dummies_df_list += [
            pd.get_dummies(ordered_cat, drop_first=False, dtype=np.int8).T
        ]

    # [1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490]
    # Concat dummy dataframes in one go:
    ## Level is constructed for free.
    roll_mat_df = pd.concat(
        dummies_df_list, keys=list(range(12)), names=["level", "id"]
    )  # .astype(np.int8, copy=False)

    # Save values as sparse matrix & save index for future reference:
    roll_index = roll_mat_df.index
    roll_mat_csr = csr_matrix(roll_mat_df.values)

    # nosemgrep
    roll_mat_df.to_pickle(data_path + "/ordered_roll_mat_df.pkl")

    del dummies_df_list, roll_mat_df
    gc.collect()

    S = get_s(roll_mat_csr, sales, prediction_start)
    W = get_w(roll_mat_csr, data[["id", "sale_usd"]])
    SW = W / np.sqrt(S)

    sw_df = pd.DataFrame(
        np.stack((S, W, SW), axis=-1),
        index=roll_index,
        columns=["s", "w", "sw"],
    )
    # nosemgrep
    sw_df.to_pickle(data_path + f"/ordered_sw_df_p{prediction_start}.pkl")

    return sales, S, W, SW, roll_mat_csr


def load_precalculated_data(
    data_path: str, prediction_start: int
) -> Tuple[Any, Any, Any, csr_matrix]:
    # Load S and W weights for WRMSSE calcualtions:
    if not os.path.exists(data_path + f"/ordered_sw_df_p{prediction_start}.pkl"):
        calculate_and_save_data(data_path, prediction_start)
    # nosemgrep
    sw_df = pd.read_pickle(
        data_path + f"/ordered_sw_df_p{prediction_start}.pkl"
    )  # nosec: [B301:blacklist]
    S = sw_df.s.values
    W = sw_df.w.values
    SW = sw_df.sw.values

    # Load roll up matrix to calcualte aggreagates:
    # nosemgrep
    roll_mat_df = pd.read_pickle(
        data_path + "/ordered_roll_mat_df.pkl"
    )  # nosec: [B301:blacklist]
    roll_mat_csr = csr_matrix(roll_mat_df.values)
    del roll_mat_df

    return S, W, SW, roll_mat_csr


def evaluate_wrmsse(
    data_path: str, prediction: Any, prediction_start: int, score_only: bool = True
) -> Any:
    # Loading data in two ways:
    # if S, W, SW are calculated in advance, load from pickle files
    # otherwise, calculate from scratch
    if os.path.isfile(
        data_path + f"/ordered_sw_df_p{prediction_start}.pkl"
    ) and os.path.isfile(data_path + "/ordered_roll_mat_df.pkl"):
        print("load precalculated data")
        # Sales quantities:
        sales = pd.read_csv(data_path + "/sales_train_evaluation.csv")
        S, W, SW, roll_mat_csr = load_precalculated_data(data_path, prediction_start)
    else:
        print("load data from scratch")
        sales, S, W, SW, roll_mat_csr = calculate_and_save_data(
            data_path, prediction_start
        )

    # Ground truth:
    dayCols = [
        f"d_{i}" for i in range(prediction_start, prediction_start + prediction_length)
    ]
    y_true = sales[dayCols]

    error = prediction - y_true.values
    results = wrmsse(error, score_only, roll_mat_csr, S, W, SW)

    return results


if __name__ == "__main__":
    DATA_DIR = "./m5-forecasting-accuracy/"
    PREDICTION_START = 1914  # 1886:offline val, 1914:validation, 1942:evaluation

    prediction_pd = pd.read_csv("submission_1672340217.csv")
    prediction: Any = np.array(prediction_pd.values[:30490, 1:], dtype=np.float32)

    # First Evaluator
    # wrmsse, aggregated_wrmsse, _, _ = evaluate_wrmsse(data_path=DATA_DIR, prediction=prediction, prediction_start=PREDICTION_START, score_only=False)
    evaluated_wrmsse = evaluate_wrmsse(
        data_path=DATA_DIR,
        prediction=prediction,
        prediction_start=PREDICTION_START,
        score_only=True,
    )
    print("---------------------------------------------------")
    print("First Evaluator")
    print("WRMSSE:", evaluated_wrmsse)
    # for i, val in enumerate(aggregated_wrmsse):
    # print(f'WRMSSE level #{i+1}: {val}')
