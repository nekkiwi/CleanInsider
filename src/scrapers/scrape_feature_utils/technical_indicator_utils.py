import pandas as pd
import yfinance as yf
import numpy as np
import os
from functools import lru_cache
from typing import Optional

# 2. Apply the monkey-patch to fix the numpy module in memory.
#    This adds the 'NaN' alias that the old pandas_ta version expects.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# 3. NOW, it is safe to import pandas_ta, as it will find the patched numpy.
import pandas_ta as ta

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
        
    # Calculate indicators using pandas_ta
    stock_df.ta.rsi(length=14, append=True)
    stock_df.ta.macd(fast=12, slow=26, signal=9, append=True)
    stock_df.ta.bbands(length=20, append=True)
    stock_df.ta.adx(length=14, append=True)
    stock_df.ta.obv(append=True)
    
    # Rename columns for clarity and to avoid special characters
    stock_df.rename(columns={
        'RSI_14': 'RSI',
        'MACD_12_26_9': 'MACD',
        'MACDh_12_26_9': 'MACD_Histogram',
        'MACDs_12_26_9': 'MACD_Signal',
        'BBL_20_2.0': 'BB_Lower',
        'BBM_20_2.0': 'BB_Mid',
        'BBU_20_2.0': 'BB_Upper',
        'ADX_14': 'ADX',
        'OBV': 'OBV'
    }, inplace=True)
    
    return stock_df

def get_techn_ind_local(ticker: str, filing_date: pd.Timestamp, config) -> pd.DataFrame:
    """
    Calculates technical indicators using the local Stooq data file path from config.

    Args:
        ticker (str): The stock ticker.
        filing_date (pd.Timestamp): The date of the filing.
        config: The central configuration module.

    Returns:
        pd.DataFrame: A single-row DataFrame with indicators.
    """
    # Get Stooq data path from the config object
    # The config defines STOOQ_DATABASE_PATH as the root directory of the
    # downloaded Stooq database. Use this value and handle a missing attribute
    # gracefully so the scraper can fall back to the yfinance method.
    stooq_path = getattr(config, "STOOQ_DATABASE_PATH", None)
    if not stooq_path.exists():
        return pd.DataFrame()
        
    all_stocks_df = pd.read_csv(stooq_path, header=0)
    all_stocks_df.rename(columns={'<TICKER>': 'Ticker', '<DTYYYYMMDD>': 'Date'}, inplace=True)
    
    stock_df = all_stocks_df[all_stocks_df['Ticker'] == ticker].copy()
    if stock_df.empty:
        return pd.DataFrame()
        
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%Y%m%d')
    stock_df.set_index('Date', inplace=True)
    stock_df.sort_index(inplace=True)
    
    # calculate_indicators is a local helper and doesn't need config
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
    start_date = end_date - pd.DateOffset(years=1) # Get 1 year of data to calculate indicators
    
    stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if stock_df.empty:
        return pd.DataFrame()
        
    # Calculate indicators
    indicators_df = calculate_indicators(stock_df)
    
    # Return the last row, which corresponds to the filing date
    return indicators_df.tail(1)


def get_technical_indicators_for_filing(
    ticker: str, filing_date: pd.Timestamp, config
) -> dict:
    """Return technical indicators for a single filing.

    The function first attempts to load locally stored stock data from the
    Stooq database defined in ``config``. If that fails to provide data, it
    falls back to downloading data from Yahoo Finance.  If both sources fail,
    an empty dictionary is returned.
    """

    # 1. Try local Stooq data
    local_df = get_techn_ind_local(ticker, filing_date, config)
    if not local_df.empty:
        # Convert the single-row dataframe into a dictionary of indicator values
        return local_df.reset_index(drop=True).iloc[0].to_dict()

    # 2. Fallback to Yahoo Finance
    yf_df = get_tech_ind_yf(ticker, filing_date)
    if not yf_df.empty:
        return yf_df.reset_index(drop=True).iloc[0].to_dict()

    # 3. If all methods fail, return an empty dict
    return {}
