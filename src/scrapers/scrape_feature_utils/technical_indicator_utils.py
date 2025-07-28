
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# 2. Apply the monkey-patch to fix the numpy module in memory.
#    This adds the 'NaN' alias that the old pandas_ta version expects.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# Import pandas_ta after patching numpy so the library can access np.NaN
import pandas_ta as ta  # noqa: E402,F401


def calculate_indicators(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a standard set of technical indicators on a stock history DataFrame.
    The input DataFrame must have a DatetimeIndex and columns: 'Open', 'High', 'Low', 'Close', 'Volume'.

    Args:
        stock_df (pd.DataFrame): DataFrame with stock price history.

    Returns:
        pd.DataFrame: DataFrame with technical indicator columns added.
    """
    if stock_df.empty:
        return pd.DataFrame()

    # Calculate a standard set of indicators using pandas_ta.  The specific
    # indicators were chosen to roughly match the features expected by the
    # rest of the pipeline.
    stock_df.ta.rsi(length=14, append=True)
    stock_df.ta.macd(fast=12, slow=26, signal=9, append=True)
    stock_df.ta.cci(length=14, append=True)
    stock_df.ta.roc(length=14, append=True)
    stock_df.ta.mfi(length=14, append=True)
    stock_df.ta.stoch(length=14, smooth_k=3, smooth_d=3, append=True)
    stock_df.ta.bbands(length=20, append=True)
    stock_df.ta.adx(length=14, append=True)
    stock_df.ta.obv(append=True)

    # Rename columns for clarity and to avoid special characters
    stock_df.rename(
        columns={
            "RSI_14": "RSI_14",
            "MACD_12_26_9": "MACD",
            "MACDh_12_26_9": "MACD_Hist",
            "MACDs_12_26_9": "MACD_Signal",
            "CCI_14": "CCI_14",
            "ROC_14": "ROC",
            "MFI_14": "MFI_14",
            "STOCHd_14_3_3": "STOCH_D",
            "BBL_20_2.0": "Bollinger_Lower",
            "ADX_14": "ADX_14",
            "OBV": "OBV",
        },
        inplace=True,
    )

    return stock_df


@lru_cache(maxsize=256)
def _load_stooq_history(ticker: str, database_root: Path) -> pd.DataFrame:
    """Load historical price data for ``ticker`` from a local Stooq database."""
    patterns = [f"{ticker.upper()}*.txt", f"{ticker.lower()}*.txt"]
    file_path = None
    for pattern in patterns:
        try:
            file_path = next(database_root.rglob(pattern))
            break
        except StopIteration:
            continue

    if file_path is None:
        return pd.DataFrame()

    dtypes = {
        0: str,
        1: str,
        2: str,
        3: str,
        4: float,
        5: float,
        6: float,
        7: float,
        8: float,
        9: float,
    }
    df = pd.read_csv(
        file_path,
        header=None,
        names=[
            "Ticker",
            "PER",
            "Date",
            "Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "OpenInt",
        ],
        dtype=dtypes,
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def get_techn_ind_local(ticker: str, filing_date: pd.Timestamp, database_root: Path) -> pd.DataFrame:
    """Return indicators from locally stored Stooq data."""
    if not database_root.exists():
        return pd.DataFrame()

    stock_df = _load_stooq_history(ticker, database_root)
    if stock_df.empty:
        return pd.DataFrame()

    indicators_df = calculate_indicators(stock_df)
    point_in_time_df = indicators_df[indicators_df.index <= pd.to_datetime(filing_date)]
    if point_in_time_df.empty:
        return pd.DataFrame()

    return point_in_time_df.tail(1)


def get_tech_ind_yf(ticker: str, filing_date: pd.Timestamp) -> pd.DataFrame:
    """
    Downloads stock data from Yahoo Finance and calculates technical indicators.

    Args:
        ticker (str): The stock ticker.
        filing_date (pd.Timestamp): The date of the filing.

    Returns:
        pd.DataFrame: A single-row DataFrame with indicators, or empty if download fails.
    """
    end_date = pd.to_datetime(filing_date)
    start_date = end_date - pd.DateOffset(
        years=1
    )  # Get 1 year of data to calculate indicators

    stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if stock_df.empty:
        return pd.DataFrame()

    # Calculate indicators
    indicators_df = calculate_indicators(stock_df)

    # Return the last row, which corresponds to the filing date
    return indicators_df.tail(1)


def get_technical_indicators_for_filing(
    ticker: str, filing_date: pd.Timestamp, database_root: Path
) -> dict:
    """Retrieve technical indicators for a single filing.

    This function first attempts to load pre-downloaded Stooq data.  If that
    fails, it falls back to downloading data from Yahoo Finance.  If both steps
    fail, an empty dictionary is returned.
    """

    # Try the fast local method first
    local_df = get_techn_ind_local(ticker, filing_date, database_root)
    if local_df.empty:
        # Fall back to yfinance
        local_df = get_tech_ind_yf(ticker, filing_date)

    if local_df.empty:
        return {}

    # Convert the resulting single-row DataFrame into a plain dictionary of
    # features.  ``iloc[0]`` is safe here because we ensured ``tail(1)`` above.
    return local_df.iloc[0].to_dict()
