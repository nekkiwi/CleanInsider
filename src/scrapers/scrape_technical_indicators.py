from pathlib import Path

import pandas as pd
from tqdm import tqdm

# **FIX: Import the single, correct function from the utils file.**
from .scrape_feature_utils.technical_indicator_utils import (
    get_technical_indicators_for_filing,
)


def scrape_all_technical_indicators(
    base_df: pd.DataFrame, output_path: Path, stooq_path: Path
) -> pd.DataFrame:
    """
    Orchestrates scraping of all technical, date, and market features using
    a central config object.
    """
    # Use only Ticker and Filing Date as unique identifiers, as Trade Date is no longer needed here.
    filings_to_process = base_df[["Ticker", "Filing Date"]].drop_duplicates()
    all_features = []

    print(
        f"Processing {len(filings_to_process)} unique Ticker/Date pairs for technical features..."
    )
    for _, row in tqdm(
        filings_to_process.iterrows(),
        total=len(filings_to_process),
        desc="Fetching Technical Features",
    ):

        # **FIX: Call the correct, new function and pass the config object.**
        features = get_technical_indicators_for_filing(
            row["Ticker"], row["Filing Date"], stooq_path
        )

        # Add identifying keys back to the feature dictionary
        features["Ticker"] = row["Ticker"]
        features["Filing Date"] = row["Filing Date"]
        all_features.append(features)

    if not all_features:
        features_df = pd.DataFrame(columns=["Ticker", "Filing Date"])
    else:
        features_df = pd.DataFrame(all_features)
        # Ensure consistent column order with Ticker and Filing Date first
        cols = features_df.columns.tolist()
        if "Ticker" in cols and "Filing Date" in cols:
            cols.insert(0, cols.pop(cols.index("Filing Date")))
            cols.insert(0, cols.pop(cols.index("Ticker")))
            features_df = features_df.reindex(columns=cols)

    features_df.to_excel(output_path, index=False)
    print(f"Stage 2 data saved to {output_path}")

    return features_df
