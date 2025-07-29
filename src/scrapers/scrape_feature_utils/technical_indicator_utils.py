from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# Monkey-patch numpy to support pandas_ta expectations
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas_ta as ta  # Ensure pandas_ta is imported after numpy patch

# **FIX: Re-introduced market data helper functions**
@lru_cache(maxsize=1)
def get_market_data(start_date, end_date):
    """Cached function to fetch SPY and VIX data for the current run."""
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=False)
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False, auto_adjust=False)
    return spy, vix

def get_market_regime_features(filing_date: pd.Timestamp) -> dict:
    """Calculates VIX and SPY market trend indicators for a given date."""
    end_date = pd.to_datetime(filing_date) + pd.Timedelta(days=1)
    start_date = end_date - pd.DateOffset(years=1)
    
    spy_df, vix_df = get_market_data(start_date, end_date)
    spy = spy_df.copy()
    vix = vix_df.copy()
    
    if spy.empty or vix.empty: return {}
    
    spy['SMA50'] = spy['Close'].rolling(window=50).mean()
    spy['SMA200'] = spy['Close'].rolling(window=200).mean()
    vix['SMA50'] = vix['Close'].rolling(window=50).mean()

    # Use .tail(1) to safely get the last row as a 1-row DataFrame
    latest_spy_row = spy.loc[:filing_date].tail(1)
    latest_vix_row = vix.loc[:filing_date].tail(1)

    # If no data exists before the filing date, return empty
    if latest_spy_row.empty or latest_vix_row.empty:
        return {}
        
    # Use .item() to extract the single scalar value, preventing the ValueError
    close_val = latest_spy_row['Close'].values[0]
    sma50_val = latest_spy_row['SMA50'].values[0]
    sma200_val = latest_spy_row['SMA200'].values[0]
    
    vix_close_val = latest_vix_row['Close'].values[0][0]
    vix_sma50_val = latest_vix_row['SMA50'].values[0]

    is_above_sma50 = (close_val > sma50_val) if pd.notna(close_val) and pd.notna(sma50_val) else False
    is_above_sma200 = (close_val > sma200_val) if pd.notna(close_val) and pd.notna(sma200_val) else False

    return {
        'VIX_Close': vix_close_val,
        'VIX_SMA50': vix_sma50_val,
        'SP500_Above_SMA50': 1 if is_above_sma50 else 0,
        'SP500_Above_SMA200': 1 if is_above_sma200 else 0
    }

def calculate_indicators(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a comprehensive set of technical indicators on a stock OHLCV DataFrame.
    Prints debug info at each step and safely handles missing or insufficient data.
    """
    # print("[calculate_indicators] Received stock_df with shape:", stock_df.shape)

    if stock_df.empty:
        # print("[calculate_indicators] Empty DataFrame, returning empty.")
        return pd.DataFrame()

    # 1. Ensure core OHLCV columns are float64
    # print("[calculate_indicators] Casting core OHLCV to float64...")
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in stock_df:
            stock_df[col] = stock_df[col].astype('float64')
    # print("[calculate_indicators] After cast shape:", stock_df.shape)

    def safe_compute(name, func, expected_cols=None):
        # print(f"[calculate_indicators] Calculating {name}...")
        try:
            result = func()
            if isinstance(result, pd.DataFrame) and expected_cols:
                for c in expected_cols:
                    if c in result:
                        stock_df[c] = result[c].astype('float64')
            elif isinstance(result, pd.Series) and expected_cols:
                stock_df[expected_cols[0]] = result.astype('float64')
        except Exception as e:
            pass
            # print(f"[calculate_indicators] {name} failed: {e}")
        # Debug sample
        if expected_cols:
            present = [c for c in expected_cols if c in stock_df]
            sample = stock_df[present].dropna()
            msg = sample.tail(3) if not sample.empty else "all values are NaN"
            # print(f"[calculate_indicators] {name} sample for {present}:\n", msg)

    # Momentum Indicators
    safe_compute("RSI", lambda: stock_df.ta.rsi(length=14), ['RSI_14'])
    safe_compute("MACD", lambda: stock_df.ta.macd(fast=12, slow=26, signal=9), ['MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9'])
    safe_compute("CCI", lambda: stock_df.ta.cci(length=14), ['CCI_14_0.015'])
    safe_compute("ROC", lambda: stock_df.ta.roc(length=14), ['ROC_14'])
    safe_compute("CMO", lambda: stock_df.ta.cmo(length=14), ['CMO_14'])
    safe_compute("AO", lambda: stock_df.ta.ao(), ['AO_5_34'])
    safe_compute("APO", lambda: stock_df.ta.apo(fast=12, slow=26), ['APO_12_26'])
    safe_compute("BOP", lambda: stock_df.ta.bop(), ['BOP'])
    safe_compute("COPPOCK", lambda: stock_df.ta.coppock(), ['COPP_10_11'])
    safe_compute("KST", lambda: stock_df.ta.kst(), ['KST_10_15_20_30','signal_9'])
    safe_compute("MOM", lambda: stock_df.ta.mom(length=10), ['MOM_10'])
    safe_compute("PPO", lambda: stock_df.ta.ppo(), ['PPO_12_26_9','PPOS_12_26_9','PPOH_12_26_9'])
    safe_compute("TRIX", lambda: stock_df.ta.trix(length=15), ['TRIX_15'])
    safe_compute("TSI", lambda: stock_df.ta.tsi(), ['TSI_25_13'])
    safe_compute("UO", lambda: stock_df.ta.uo(), ['UO'])
    safe_compute("WILLR", lambda: stock_df.ta.willr(), ['WILLR_14'])

    # Overlap Indicators
    safe_compute("EMA", lambda: stock_df.ta.ema(length=20), ['EMA_20'])
    safe_compute("SMA", lambda: stock_df.ta.sma(length=20), ['SMA_20'])
    safe_compute("DEMA", lambda: stock_df.ta.dema(length=20), ['DEMA_20'])
    safe_compute("TEMA", lambda: stock_df.ta.tema(length=20), ['TEMA_20'])
    safe_compute("WMA", lambda: stock_df.ta.wma(length=20), ['WMA_20'])
    safe_compute("ZLMA", lambda: stock_df.ta.zlma(length=20), ['ZLMA_20'])
    safe_compute("HL2", lambda: stock_df.ta.hl2(), ['HL2'])
    safe_compute("HLC3", lambda: stock_df.ta.hlc3(), ['HLC3'])
    safe_compute("OHLC4", lambda: stock_df.ta.ohlc4(), ['OHLC4'])
    safe_compute("VWAP", lambda: stock_df.ta.vwap(), ['VWAP'])
    safe_compute("VWMA", lambda: stock_df.ta.vwma(length=20), ['VWMA_20'])
    safe_compute("HMA", lambda: stock_df.ta.hma(length=20), ['HMA_20'])
    # Ichimoku returns DataFrame of 5 lines
    safe_compute("ICHIMOKU", lambda: stock_df.ta.ichimoku(), ['ICHIMOKU_A_9_26_52','ICHIMOKU_B_9_26_52'])

    # Performance Indicators
    safe_compute("LOG_RETURN", lambda: stock_df.ta.log_return(cumulative=False), ['LOGRET'])
    safe_compute("PERCENT_RETURN", lambda: stock_df.ta.percent_return(cumulative=False), ['PCTRET'])

    # Statistical Indicators
    safe_compute("KURTOSIS", lambda: stock_df.ta.kurtosis(), ['KURTOSIS'])
    safe_compute("MAD", lambda: stock_df.ta.mad(), ['MAD'])
    safe_compute("MEDIAN", lambda: stock_df.ta.median(), ['MEDIAN'])
    safe_compute("QUANTILE", lambda: stock_df.ta.quantile(q=0.5), ['QUANTILE_0.5'])
    safe_compute("SKEW", lambda: stock_df.ta.skew(), ['SKEW'])
    safe_compute("STDEV", lambda: stock_df.ta.stdev(), ['STDEV'])
    safe_compute("VARIANCE", lambda: stock_df.ta.variance(), ['VAR'])
    safe_compute("ZSCORE", lambda: stock_df.ta.zscore(), ['ZSCORE'])

    # Trend Indicators
    safe_compute("ADX", lambda: stock_df.ta.adx(), ['ADX_14','DMP_14','DMN_14'])
    safe_compute("AROON", lambda: stock_df.ta.aroon(), ['AROON_UP_25','AROON_DN_25','AROON_OSC_25'])
    safe_compute("DPO", lambda: stock_df.ta.dpo(), ['DPO_20'])
    safe_compute("QSTICK", lambda: stock_df.ta.qstick(), ['QSTICK_7'])
    safe_compute("VORTEX", lambda: stock_df.ta.vortex(), ['VORTX_pos_14','VORTX_neg_14','VORTXOsc_14'])
    safe_compute("DECREASING", lambda: stock_df.ta.decreasing(length=3), ['DEC_3'])
    safe_compute("INCREASING", lambda: stock_df.ta.increasing(length=3), ['INC_3'])

    # Volatility Indicators
    safe_compute("ATR", lambda: stock_df.ta.atr(), ['ATR_14'])
    safe_compute("BBANDS", lambda: stock_df.ta.bbands(), ['BBL_20_2.0','BBM_20_2.0','BBU_20_2.0','BBB_20_2.0','BBP_20_2.0'])
    safe_compute("ACCBANDS", lambda: stock_df.ta.accbands(), ['ACCBL_5_2_2','ACCBM_5_2_2','ACCBU_5_2_2'])
    safe_compute("DONCHIAN", lambda: stock_df.ta.donchian(), ['DCL_20', 'DCM_20', 'DCU_20'])
    safe_compute("KC", lambda: stock_df.ta.kc(), ['KCL_20_2.0', 'KCM_20_2.0', 'KCU_20_2.0'])
    safe_compute("MASSI", lambda: stock_df.ta.massi(), ['MASSI_25_9'])
    safe_compute("NATR", lambda: stock_df.ta.natr(), ['NATR_14'])
    safe_compute("TRUE_RANGE", lambda: stock_df.ta.true_range(), ['TR'])

    # Volume Indicators
    safe_compute("AD", lambda: stock_df.ta.ad(), ['AD'])
    safe_compute("ADOSC", lambda: stock_df.ta.adosc(), ['ADOSC_3_10'])
    safe_compute("CMF", lambda: stock_df.ta.cmf(), ['CMF_20'])
    safe_compute("EFI", lambda: stock_df.ta.efi(), ['EFI_13'])
    safe_compute("EOM", lambda: stock_df.ta.eom(), ['EOM_14'])
    # safe_compute("MFI", lambda: stock_df.ta.mfi(), ['MFI_14'])
    safe_compute("NVI", lambda: stock_df.ta.nvi(), ['NVI_14'])
    safe_compute("OBV", lambda: stock_df.ta.obv(), ['OBV'])
    safe_compute("PVI", lambda: stock_df.ta.pvi(), ['PVI_14'])
    safe_compute("PVT", lambda: stock_df.ta.pvt(), ['PVT'])
    safe_compute("VP", lambda: stock_df.ta.vp(), ['VP_10'])
    safe_compute("PVOL", lambda: stock_df.ta.pvol(), ['PVOL_14'])

    # 3. Rename for consistency
    # print("[calculate_indicators] Renaming columns...")
    stock_df.rename(columns={
        'MACD_12_26_9':'MACD','MACDh_12_26_9':'MACD_Hist','MACDs_12_26_9':'MACD_Signal',
        'CCI_14_0.015':'CCI_14','ROC_14':'ROC','STOCHk_14_3_3':'STOCH_K','STOCHd_14_3_3':'STOCH_D',
        'BBL_20_2.0':'Bollinger_Lower','BBM_20_2.0':'Bollinger_Mid','BBU_20_2.0':'Bollinger_Upper',
        'BBB_20_2.0':'Bollinger_BandWidth','BBP_20_2.0':'Bollinger_Percent','DMP_14':'+DI','DMN_14':'-DI'
    }, inplace=True)
    # print("[calculate_indicators] Final shape:", stock_df.shape)
    return stock_df

@lru_cache(maxsize=256)
def _load_stooq_history(ticker: str, database_root: Path) -> pd.DataFrame:
    """
    Load historical OHLCV data from a local Stooq .txt, printing each step.
    """
    # print(f"[_load_stooq_history] Searching for files for ticker {ticker} in {database_root}")
    patterns = [f"{ticker.upper()}*.txt", f"{ticker.lower()}*.txt"]
    file_path = None
    for pattern in patterns:
        # print(f"[_load_stooq_history] Trying pattern: {pattern}")
        try:
            file_path = next(database_root.rglob(pattern))
            # print(f"[_load_stooq_history] Found file: {file_path}")
            break
        except StopIteration:
            continue
    if file_path is None:
        # print("[_load_stooq_history] No file found, returning empty.")
        return pd.DataFrame()

    dtypes = {0: str, 1: str, 2: str, 3: str,
              4: float, 5: float, 6: float, 7: float, 8: float, 9: float}
    # print(f"[_load_stooq_history] Reading CSV from {file_path}")
    df = pd.read_csv(
        file_path,
        header=None,
        names=["Ticker","PER","Date","Time","Open","High","Low","Close","Volume","OpenInt"],
        dtype=dtypes,
        skiprows=1,
        on_bad_lines="skip",
    )
    # print("[_load_stooq_history] After read, shape:", df.shape)
    # print(df.head())

    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    # print("[_load_stooq_history] After date parse, null dates:", df["Date"].isna().sum())
    df.dropna(subset=["Date"], inplace=True)
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    # print("[_load_stooq_history] Final OHLCV shape:", df.shape)

    return df[["Open","High","Low","Close","Volume"]]


def get_techn_ind_local(ticker: str, filing_date: pd.Timestamp, database_root: Path) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Retrieve technical indicators locally."""
    if not database_root.exists():
        return pd.DataFrame(), None
    stock_df = _load_stooq_history(ticker, database_root)
    if stock_df.empty:
        return pd.DataFrame(), None
    
    # **ADD: Get the IPO date from the full history**
    ipo_date = stock_df.index.min()
    
    indicators_df = calculate_indicators(stock_df)
    point_in_time = indicators_df[indicators_df.index <= pd.to_datetime(filing_date)]
    if point_in_time.empty:
        return pd.DataFrame(), None
        
    result = point_in_time.tail(1)
    # **FIX: Return the result DataFrame and the IPO date**
    return result, ipo_date

# **FIX: This function now returns a tuple (DataFrame, ipo_date)**
def get_tech_ind_yf(ticker: str, filing_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Fallback to Yahoo Finance if local fails."""
    end_date = pd.to_datetime(filing_date) + pd.Timedelta(days=1)
    start_date = end_date - pd.DateOffset(years=2) # Use 2 years for more robust indicators
    stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    if stock_df.empty:
        return pd.DataFrame(), None
        
    # **ADD: Get the IPO date from the full history**
    ipo_date = stock_df.index.min()
    
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)
    if 'Adj Close' in stock_df.columns:
        stock_df = stock_df.drop(columns=['Adj Close'])
    try:
        stock_df = stock_df[['Open','High','Low','Close','Volume']]
    except KeyError:
        return pd.DataFrame(), None
    stock_df = stock_df.apply(pd.to_numeric, errors='coerce').dropna()
    if stock_df.empty:
        return pd.DataFrame(), None
        
    indicators_df = calculate_indicators(stock_df)
    point_in_time = indicators_df[indicators_df.index <= pd.to_datetime(filing_date)]
    if point_in_time.empty:
        return pd.DataFrame(), None
        
    result = point_in_time.tail(1)
    # **FIX: Return the result DataFrame and the IPO date**
    return result, ipo_date


def get_technical_indicators_for_filing(ticker: str, filing_date: pd.Timestamp, database_root: Path) -> dict:
    """
    Try local Stooq data first; if that returns empty, fall back to Yahoo Finance.
    Also calculates and includes the 'Days_Since_IPO' feature.
    """
    # Attempt local data
    local_df, ipo_date = get_techn_ind_local(ticker, filing_date, database_root)
    if not local_df.empty:
        result = local_df.iloc[0].to_dict()
        # **ADD: IPO logic**
        if ipo_date:
            result['Days_Since_IPO'] = (pd.to_datetime(filing_date) - ipo_date).days
        market_features = get_market_regime_features(filing_date)
        result.update(market_features)
        return result
        
    # Fallback to Yahoo
    yf_df, ipo_date = get_tech_ind_yf(ticker, filing_date)
    if not yf_df.empty:
        result = yf_df.iloc[0].to_dict()
        # **ADD: IPO logic**
        if ipo_date:
            result['Days_Since_IPO'] = (pd.to_datetime(filing_date) - ipo_date).days
        
        market_features = get_market_regime_features(filing_date)
        result.update(market_features)
        return result
        
    # Both failed
    return {}