# file: src/preprocess/time_based_splitter.py

import numpy as np
import pandas as pd


def create_time_based_splits(
    df: pd.DataFrame, n_splits: int = 7
) -> (pd.DataFrame, dict):
    """
    Sorts the DataFrame by date and divides it into a specified number of time-based splits.
    It also generates a dictionary mapping splits to their Ticker-FilingDate combinations.

    Args:
        df (pd.DataFrame): The input DataFrame, must contain 'Filing Date' and 'Ticker'.
        n_splits (int): The number of portions to split the data into.

    Returns:
        pd.DataFrame: The DataFrame with an added 'split_id' column.
        dict: A JSON-serializable dictionary with the Ticker-FilingDate mapping.
    """
    print(f"\n--- Creating {n_splits} Time-Based Splits ---")
    if "Filing Date" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'Filing Date' column for time-based splitting."
        )

    df["Filing Date"] = pd.to_datetime(df["Filing Date"])
    df = df.sort_values("Filing Date").reset_index(drop=True)

    rows_per_split = len(df) // n_splits
    split_assignments = np.repeat(np.arange(1, n_splits + 1), rows_per_split)

    remainder = len(df) - len(split_assignments)
    if remainder > 0:
        split_assignments = np.concatenate(
            [split_assignments, np.full(remainder, n_splits)]
        )

    df["split_id"] = split_assignments

    split_ticker_map = {f"split_{i}": {} for i in range(1, n_splits + 1)}
    for split_id, group in df.groupby("split_id"):
        for _, row in group.iterrows():
            ticker = row["Ticker"]
            filing_date = row["Filing Date"].strftime("%Y-%m-%d")

            if ticker not in split_ticker_map[f"split_{split_id}"]:
                split_ticker_map[f"split_{split_id}"][ticker] = []
            split_ticker_map[f"split_{split_id}"][ticker].append(filing_date)

    print(f"✅ Data sorted and split into {n_splits} portions.")
    return df, split_ticker_map
