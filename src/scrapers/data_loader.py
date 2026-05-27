# file: src/scrapers/data_loader.py

from functools import lru_cache
from pathlib import Path

import pandas as pd
import yfinance as yf

FINAL_COLS = ["Open", "High", "Low", "Close", "Volume"]


def read_csv_safe(filepath):
    """Safely read CSV files, handling empty/corrupted files gracefully."""
    try:
        if Path(filepath).stat().st_size == 0:
            # print(f"  [READ-WARN] Empty file detected: {filepath.name}")
            return pd.DataFrame()
        df = pd.read_csv(filepath)
        # print(f"  [READ-SUCCESS] Loaded file {filepath.name} with shape {df.shape}")
        # print(f"  [READ-DEBUG] Columns in {filepath.name}: {df.columns.tolist()}")
        # if not df.empty:
        # print(f"  [READ-DEBUG] Sample data from {filepath.name}:")
        # print(f"    {df.head(2).to_string()}")
        return df
    except pd.errors.EmptyDataError:
        # print(f"  [READ-ERROR] EmptyDataError reading file: {filepath.name}")
        return pd.DataFrame()
    except Exception:
        # print(f"  [READ-ERROR] Error reading file {filepath.name}: {e}")
        return pd.DataFrame()


def _standardize_and_clean(df: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    """
    A single, robust function to clean any OHLCV dataframe. This is the
    definitive version designed to handle all known edge cases.
    """
    # print(f"  [CLEAN-START] Processing ticker {ticker} from {source}")
    # print(f"  [CLEAN-DEBUG] Input data shape: {df.shape}")

    if df.empty:
        # print(f"  [CLEAN-WARN] Empty DataFrame received for {ticker} from {source}")
        return pd.DataFrame()

    # print(f"  [CLEAN-DEBUG] Input columns: {df.columns.tolist()}")

    df_clean = df.copy()

    # --- Step 1: Handle Column Structure ---
    # First, handle yfinance's MultiIndex columns if they exist.
    if isinstance(df_clean.columns, pd.MultiIndex):
        # print(f"  [CLEAN-DEBUG] Detected MultiIndex columns for {ticker}")
        df_clean.columns = df_clean.columns.get_level_values(0)
        # print(f"  [CLEAN-DEBUG] Flattened columns: {df_clean.columns.tolist()}")

    # Aggressively standardize all column names to simple, flat strings.
    df_clean.columns = [
        str(col).lower().replace("<", "").replace(">", "").strip()
        for col in df_clean.columns
    ]
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated(keep="first")]

    # --- Step 2: Unify Date into the Index (THE CRITICAL FIX) ---
    # If 'date' exists as a column (from local files), set it as the index.
    if "date" in df_clean.columns:
        # Stooq local files store the date as a YYYYMMDD integer; pd.to_datetime
        # would misread that as epoch-nanoseconds (-> 1970). Parse with an explicit
        # format, falling back to generic parsing for any other source.
        date_str = df_clean["date"].astype(str).str.replace(r"[-/]", "", regex=True)
        parsed = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(df_clean["date"], errors="coerce")
        df_clean["date"] = parsed
        # print(f"  [CLEAN-DEBUG] {date_na_count} rows had invalid dates and will be dropped")
        # Remove rows where date parsing failed before setting index
        df_clean = df_clean[~df_clean["date"].isna()]
        if not df_clean.empty:
            df_clean.set_index("date", inplace=True)
            # print(f"  [CLEAN-DEBUG] Set date as index. Date range: {df_clean.index.min()} to {df_clean.index.max()}")
    # If the index isn't already a DatetimeIndex (from yfinance), convert it.
    elif not isinstance(df_clean.index, pd.DatetimeIndex):
        # print(f"  [CLEAN-DEBUG] Converting existing index to DatetimeIndex for {ticker}")
        df_clean.index = pd.to_datetime(df_clean.index, errors="coerce")
        # Drop any rows whose index could not be parsed as a date
        # print(f"  [CLEAN-DEBUG] {index_na_count} rows had invalid index dates and will be dropped")
        df_clean = df_clean[~df_clean.index.isna()]
        # if not df_clean.empty:
        # print(f"  [CLEAN-DEBUG] Date range after index conversion: {df_clean.index.min()} to {df_clean.index.max()}")

    # Now that the index is the date, standardize its name
    df_clean.index.name = "Date"

    if df_clean.empty:
        # print(f"  [CLEAN-WARN] DataFrame became empty after date processing for {ticker}")
        return pd.DataFrame()

    # print(f"  [CLEAN-DEBUG] Shape after date processing: {df_clean.shape}")

    # --- Step 3: Standardize OHLCV Data ---
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "vol": "Volume",
        "volume": "Volume",
        "adj close": "Adj Close",  # Handle adjusted close if present
    }
    df_clean.rename(columns=rename_map, inplace=True)
    # print(f"  [CLEAN-DEBUG] Columns after OHLCV rename: {df_clean.columns.tolist()}")

    # Check if we have the required columns - BE MORE LENIENT
    existing_cols = [col for col in FINAL_COLS if col in df_clean.columns]
    # print(f"  [CLEAN-DEBUG] Available OHLCV columns for {ticker}: {existing_cols}")

    if len(existing_cols) < 4:  # Allow missing volume if necessary
        # print(f"  [CLEAN-ERROR] Not enough OHLCV columns for {ticker} from {source}. Need at least 4, have {len(existing_cols)}")
        return pd.DataFrame()

    df_final = df_clean[existing_cols].copy()

    # Convert all data to numeric, coercing errors to NaN
    for col in existing_cols:
        df_final[col].notna().sum()
        df_final[col] = pd.to_numeric(df_final[col], errors="coerce")
        df_final[col].notna().sum()
        # if before_conversion != after_conversion:
        # print(f"  [CLEAN-DEBUG] Column {col}: {before_conversion - after_conversion} values became NaN during conversion")

    # Only require Close to be valid, be more lenient with other columns
    df_final["Close"].isna().sum()
    # print(f"  [CLEAN-DEBUG] {close_na_count} rows have NaN Close values and will be dropped")
    df_final = df_final.dropna(subset=["Close"])

    # --- Step 4: Detect and Apply Split Adjustments (THE DEFINITIVE FIX) ---
    # This ensures that data from any source is properly adjusted.
    if "Adj Close" not in df_final.columns:
        close_to_prev_close_ratio = df_final["Close"] / df_final["Close"].shift(1)
        # Detect splits (e.g., a 50% price drop is a 2-for-1 split, ratio ~0.5)
        # We look for large drops, typical of 2:1, 3:1, or 4:1 splits.
        split_candidates = close_to_prev_close_ratio[
            (close_to_prev_close_ratio > 0.1) & (close_to_prev_close_ratio < 0.7)
        ]

        for date, ratio in split_candidates.items():
            # Round to the nearest common split ratio (e.g., 0.5, 0.33, 0.25)
            split_ratio = 1 / round(1 / ratio)

            # Both sources (local Stooq, yfinance auto_adjust=True) are already
            # split-adjusted, so this should rarely fire. Only treat a drop as a
            # split when it's within 2% of a clean 1/N ratio; this keeps real
            # 2:1/3:1/4:1 splits but rejects genuine single-day crashes (common
            # for the small/penny-cap insider universe) that would otherwise
            # corrupt all prior prices.
            if abs(ratio - split_ratio) > 0.02:
                continue

            # Adjust all prices and volume before this date
            price_cols = ["Open", "High", "Low", "Close"]
            df_final.loc[df_final.index < date, price_cols] *= split_ratio
            if "Volume" in df_final.columns:
                df_final.loc[df_final.index < date, "Volume"] = (
                    (df_final.loc[df_final.index < date, "Volume"] / split_ratio)
                    .round()
                    .astype("int64")
                )

    # Fill missing volume with 0 if Volume column exists but has NaN values
    if "Volume" in df_final.columns:
        volume_na_count = df_final["Volume"].isna().sum()
        if volume_na_count > 0:
            df_final["Volume"] = df_final["Volume"].fillna(0)

    if df_final.empty:
        # print(f"  [CLEAN-ERROR] Final DataFrame is empty after all cleaning for {ticker} from {source}")
        return pd.DataFrame()

    # print(f"  [CLEAN-SUCCESS] Final data for {ticker} from {source}:")
    # print(f"    Shape: {df_final.shape}")
    # print(f"    Date range: {df_final.index.min().date()} to {df_final.index.max().date()}")
    # print(f"    Sample data:")
    # print(f"    {df_final.head(2).to_string()}")

    return df_final.sort_index()


def _load_from_yfinance(
    ticker: str, required_start_date: pd.Timestamp = None
) -> pd.DataFrame:
    """Load + clean OHLCV from yfinance. Returns empty DataFrame on failure."""
    start_date = required_start_date
    if start_date is not None:
        start_date = start_date - pd.Timedelta(days=30)  # buffer
    try:
        data = yf.download(
            ticker, start=start_date, progress=False, timeout=15, auto_adjust=True
        )
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    cleaned = _standardize_and_clean(data, ticker, source="yfinance")
    # Require a reasonable amount of data to consider it a good hit.
    return cleaned if (not cleaned.empty and len(cleaned) > 10) else pd.DataFrame()


def _load_from_local(ticker: str, db_path_str: str) -> pd.DataFrame:
    """Load + clean OHLCV from the local Stooq DB. Returns empty on miss."""
    db_path = Path(db_path_str)
    if not db_path.exists():
        return pd.DataFrame()
    ticker_lower = ticker.lower()
    # Stooq files are "<ticker>.<country>.txt" (e.g. aapl.us.txt, a.us.txt). Match
    # the ticker segment EXACTLY. The old code compared Path.stem, which keeps the
    # country suffix ("a.us" != "a"), so it never matched and fell back to the first
    # of many rglob hits — loading the WRONG ticker's prices for short/substring
    # tickers (A, AA, C, ...). Now: require an exact first-segment match; if none,
    # return empty so yfinance serves it. Never load a wrong ticker's data.
    # Also try the dash form for class shares (OpenInsider "BRK.B" -> Stooq "brk-b").
    variants = [ticker_lower]
    if "." in ticker_lower:
        variants.append(ticker_lower.replace(".", "-"))
    for v in variants:
        matches = [
            f for f in db_path.rglob(f"{v}.*txt") if f.name.lower().split(".")[0] == v
        ]
        if matches:
            local_data = read_csv_safe(matches[0])
            if not local_data.empty:
                return _standardize_and_clean(local_data, ticker, source="local_file")
    return pd.DataFrame()


@lru_cache(maxsize=None)
def _load_ohlcv_cached(
    ticker: str,
    db_path_str: str,
    required_start_date: pd.Timestamp,
    prefer_local: bool,
    local_only: bool,
) -> pd.DataFrame:
    """Cached core: resolve source order and return the first non-empty frame."""
    if local_only:
        sources = [lambda: _load_from_local(ticker, db_path_str)]
    elif prefer_local:
        sources = [
            lambda: _load_from_local(ticker, db_path_str),
            lambda: _load_from_yfinance(ticker, required_start_date),
        ]
    else:
        sources = [
            lambda: _load_from_yfinance(ticker, required_start_date),
            lambda: _load_from_local(ticker, db_path_str),
        ]
    for load in sources:
        df = load()
        if not df.empty:
            return df
    return pd.DataFrame()


def load_ohlcv_with_fallback(
    ticker: str,
    db_path_str: str,
    required_start_date: pd.Timestamp = None,
    prefer_local: bool = None,
    local_only: bool = None,
) -> pd.DataFrame:
    """
    Load OHLCV with a local-Stooq / yfinance pair, ordered by preference.

    With a complete local Stooq DB, prefer_local=True (config.PREFER_LOCAL_OHLCV)
    makes bulk historical scraping fast and offline; yfinance is the fallback for
    tickers missing locally. When the local DB is absent (e.g. CI), the local
    attempt simply misses and yfinance serves the data.

    local_only=True (config.OHLCV_LOCAL_ONLY) drops the yfinance fallback entirely.
    Use for bulk historical scraping: tickers missing from Stooq (mostly delisted)
    return empty immediately instead of incurring a per-ticker yfinance timeout +
    rate-limit, which is the dominant cost of a full scrape. Live inference leaves
    this off so the small recent-ticker set can still fall back to yfinance.

    Resolves config-driven defaults here (so cache keys are concrete) and returns a
    COPY of the cached frame, so callers may mutate it without poisoning the cache.
    """
    from src import config

    if prefer_local is None:
        prefer_local = config.PREFER_LOCAL_OHLCV
    if local_only is None:
        local_only = config.OHLCV_LOCAL_ONLY

    df = _load_ohlcv_cached(
        ticker, db_path_str, required_start_date, prefer_local, local_only
    )
    return df.copy()
