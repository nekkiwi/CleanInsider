# file: src/scrapers/feature_scraper/load_annual_statements.py

import time
import warnings

import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed
from tqdm import tqdm

# Ignore common warnings from yfinance and pandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_annual_data_for_ticker(ticker_symbol: str) -> pd.Series | None:
    """Fetches and processes annual data for a single ticker."""
    try:
        # A small sleep is respectful to the API when running many requests
        time.sleep(0.01)
        ticker = yf.Ticker(ticker_symbol)

        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow

        if (
            financials.shape[1] < 2
            or balance_sheet.shape[1] < 2
            or cash_flow.shape[1] < 2
        ):
            return None

        all_statements = pd.concat([financials, balance_sheet, cash_flow])
        all_statements = all_statements.loc[
            ~all_statements.index.duplicated(keep="first")
        ]

        recent_data = all_statements.iloc[:, :2]
        recent_data.columns = ["Y1", "Y2"]

        features_y1 = recent_data["Y1"].rename(lambda x: f"FIN_{x}_Y1")
        features_y2 = recent_data["Y2"].rename(lambda x: f"FIN_{x}_Y2")
        features_diff = (recent_data["Y1"] - recent_data["Y2"]).rename(
            lambda x: f"FIN_{x}_diff_Y1_Y2"
        )

        final_features = pd.concat([features_y1, features_y2, features_diff])
        final_features["Ticker"] = ticker_symbol

        return final_features

    except Exception:
        return None


def _process_ticker_batch_in_parallel(ticker_batch: list[str]) -> list[pd.Series]:
    """
    This is the core parallel worker function. It takes a BATCH of tickers
    and uses joblib to process them all in parallel.
    """
    n_jobs = -2  # Use all available cores for the tickers within this batch

    tasks = [delayed(get_annual_data_for_ticker)(ticker) for ticker in ticker_batch]

    # Run the parallel job for this single batch
    batch_results = Parallel(n_jobs=n_jobs)(
        tqdm(tasks, desc="Processing batch", leave=False)
    )

    # Filter out any None results from failed ticker lookups
    return [res for res in batch_results if res is not None]


def generate_annual_statements(
    base_df: pd.DataFrame,
    output_path: str,
    batch_size: int = 100,
    missing_thresh: float = 0.8,
):
    """
    Orchestrates the download of annual financial data by processing
    batches sequentially, but processing tickers within each batch in parallel.
    """
    print(
        "\n--- Generating Annual Statement Features (Sequential Batches, Parallel Tickers) ---"
    )
    unique_tickers = base_df["Ticker"].unique().tolist()

    ticker_batches = [
        unique_tickers[i : i + batch_size]
        for i in range(0, len(unique_tickers), batch_size)
    ]
    print(
        f"  > Divided {len(unique_tickers)} tickers into {len(ticker_batches)} batches of size {batch_size}."
    )

    all_rows = []

    # --- THIS IS THE FIX ---
    # Use a standard for loop to iterate through the batches sequentially.
    # The outer tqdm tracks the progress of the BATCHES.
    for batch in tqdm(ticker_batches, desc="Processing Batches Sequentially"):
        # Call the parallel processing function for the current batch
        batch_results = _process_ticker_batch_in_parallel(batch)
        all_rows.extend(batch_results)
    # --- END OF FIX ---

    if not all_rows:
        print(
            "❌ No valid annual statement data could be fetched. An empty file will be created."
        )
        pd.DataFrame().to_parquet(output_path, index=False)
        return

    annual_statements_df = pd.DataFrame(all_rows)

    # --- NEW: Pre-filter sparse columns ---
    print(f"  > Original component shape: {annual_statements_df.shape}")
    missing_proportions = annual_statements_df.isnull().sum() / len(
        annual_statements_df
    )
    cols_to_drop = missing_proportions[missing_proportions >= missing_thresh].index
    annual_statements_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    print(
        f"  > Dropped {len(cols_to_drop)} columns with >= {missing_thresh:.0%} missing values."
    )
    print(f"  > Final component shape: {annual_statements_df.shape}")
    # --- END NEW ---

    annual_statements_df.to_parquet(output_path, index=False)
    print(
        f"✅ Saved {len(annual_statements_df)} records with financial data to {output_path}"
    )
