# file: src/scrapers/feature_scraper/scrape_openinsider.py

import datetime
import time
from io import StringIO
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from tqdm import tqdm
from src import config
from .feature_scraper_util.general_utils import add_date_features

def _parse_insider_titles(df: pd.DataFrame) -> pd.DataFrame:
    # print(f"[TITLES-DEBUG] Input shape: {df.shape}")
    if "Title" not in df.columns:
        print("[TITLES-WARN] No Title column found, creating default role flags")
        for role in ["CEO", "CFO", "Pres", "VP", "Dir", "TenPercent"]:
            df[role] = 0
        return df
    
    df["Title_lower"] = df["Title"].str.lower()
    df["CEO"] = df["Title_lower"].str.contains("ceo|chief executive officer", na=False).astype(int)
    df["CFO"] = df["Title_lower"].str.contains("cfo|chief financial officer", na=False).astype(int)
    df["Pres"] = df["Title_lower"].str.contains("pres|president", na=False).astype(int)
    df["VP"] = df["Title_lower"].str.contains("vp|vice president", na=False).astype(int)
    df["Dir"] = df["Title_lower"].str.contains("dir|director", na=False).astype(int)
    df["TenPercent"] = df["Title_lower"].str.contains("10%|ten percent", na=False).astype(int)
    
    df.drop(columns=["Title_lower", "Title"], inplace=True)
    # print(f"[TITLES-DEBUG] Output shape after adding role flags: {df.shape}")
    # print(f"[TITLES-DEBUG] Role distribution - CEO: {df['CEO'].sum()}, CFO: {df['CFO'].sum()}, Dir: {df['Dir'].sum()}")
    return df

def _aggregate_daily_trades(df: pd.DataFrame) -> pd.DataFrame:
    # print(f"[AGG-DEBUG] Input shape for aggregation: {df.shape}")
    # print(f"[AGG-DEBUG] Input columns: {df.columns.tolist()}")
    
    agg_funcs = {
        "Value": "sum", "Qty": "sum", "Owned": "last", "dOwn": "last",
        "CEO": "max", "CFO": "max", "Pres": "max", "VP": "max", 
        "Dir": "max", "TenPercent": "max", "Days_Since_Trade": "mean",
    }
    
    grouped = df.groupby(["Ticker", pd.Grouper(key="Filing Date", freq="D")])
    df_agg = grouped.agg(agg_funcs)
    
    if "Price" in df.columns and df["Price"].notna().any():
        df_agg["Price"] = grouped.apply(
            lambda x: np.average(x["Price"], weights=x["Value"]), include_groups=False
        )
        # print(f"[AGG-DEBUG] Added weighted average Price column")
    else:
        df_agg["Price"] = np.nan
        # print(f"[AGG-DEBUG] No Price data available, filled with NaN")
    
    df_agg["Number_of_Purchases"] = grouped.size()
    df_agg = df_agg.reset_index()
    
    return df_agg

def _scrape_date_range_worker(date_range: tuple, request_header: dict) -> pd.DataFrame | None:
    start_date, end_date = date_range
    # print(f"[SCRAPE-WORKER] Processing range {start_date.date()} to {end_date.date()}")
    
    base_url = "http://openinsider.com/screener?"
    all_data_for_range = []
    page = 1
    date_str = f"{start_date.month}%2F{start_date.day}%2F{start_date.year}+-+{end_date.month}%2F{end_date.day}%2F{end_date.year}"

    while True:
        url = f"{base_url}pl=1&ph=&ll=&lh=&fd=-1&fdr={date_str}&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&vl=10&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page={page}"
        try:
            response = requests.get(url, headers=request_header, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"class": "tinytable"})
            if table is None:
                # print(f"[SCRAPE-WORKER] No table found on page {page}, stopping")
                break
            df = pd.read_html(StringIO(str(table)))[0]
            if df.empty:
                # print(f"[SCRAPE-WORKER] Empty dataframe on page {page}, stopping")
                break
            # print(f"[SCRAPE-WORKER] Scraped page {page}: {df.shape[0]} rows")
            all_data_for_range.append(df)
            if len(df) < 1000:
                # print(f"[SCRAPE-WORKER] Page {page} had < 1000 rows, assuming last page")
                break
            page += 1
            time.sleep(0.25)
        except requests.RequestException as e:
            break

    full_df = pd.concat(all_data_for_range, ignore_index=True)
    full_df.columns = full_df.columns.str.replace("\xa0", " ", regex=False)
    # print(f"[SCRAPE-WORKER] Combined range data: {full_df.shape}")
    return full_df

def scrape_openinsider(num_weeks: int) -> pd.DataFrame:
    print(f"\n[MAIN-INFO] Starting scrape_openinsider for {num_weeks} weeks")
    
    # Step 1: Create date ranges
    end_date = datetime.datetime.now()# - datetime.timedelta(days=30*3)
    date_ranges = []
    for week in range(num_weeks):
        start_date = end_date - datetime.timedelta(days=7)
        date_ranges.append((start_date, end_date))
        # # print(f"[MAIN-DEBUG] Week {week+1}: {start_date.date()} to {end_date.date()}")
        end_date = start_date

    print(f"[MAIN-INFO] Created {len(date_ranges)} date ranges for parallel scraping")

    # Step 2: Parallel scraping
    print(f"Initiating parallel scrape for {num_weeks} weeks of data...")
    tasks = [delayed(_scrape_date_range_worker)(dr, config.REQUESTS_HEADER) for dr in date_ranges]
    results = Parallel(n_jobs=-2)(tqdm(tasks, desc="Scraping weekly data"))

    all_data_frames = [df for df in results if df is not None]
    print(f"[MAIN-INFO] Retrieved {len(all_data_frames)} successful dataframes from {len(date_ranges)} attempts")
    
    if not all_data_frames:
        print("[MAIN-ERROR] No data was scraped. Exiting.")
        return pd.DataFrame()

    df = pd.concat(all_data_frames, ignore_index=True)
    print(f"[MAIN-INFO] Raw concatenated data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    initial_shape = df.shape
    df = df.drop_duplicates()
    duplicates_removed = initial_shape[0] - df.shape[0]
    print(f"[MAIN-INFO] Removed {duplicates_removed} duplicate rows, shape now: {df.shape}")

    # Step 3: Data Cleaning
    print(f"[MAIN-INFO] Starting data cleaning phase")
    # print(f"[MAIN-DEBUG] Available columns: {df.columns.tolist()}")
    
    # Column renaming
    if "ΔOwn" in df.columns:
        df.rename(columns={"ΔOwn": "dOwn"}, inplace=True)
        # print("[MAIN-DEBUG] Renamed ΔOwn to dOwn")

    # Drop rows with missing Ticker
    before_ticker_drop = len(df)
    df.dropna(subset=["Ticker"], inplace=True)
    ticker_dropped = before_ticker_drop - len(df)
    # print(f"[MAIN-DEBUG] Dropped {ticker_dropped} rows due to missing Ticker")

    # Date conversion
    # print("[MAIN-DEBUG] Converting date columns...")
    df["Filing Date"] = pd.to_datetime(df["Filing Date"], errors="coerce")
    df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
    
    before_date_drop = len(df)
    df.dropna(subset=["Filing Date", "Trade Date"], inplace=True)
    date_dropped = before_date_drop - len(df)
    # print(f"[MAIN-DEBUG] Dropped {date_dropped} rows due to invalid dates")

    # Numeric conversion
    # print("[MAIN-DEBUG] Converting numeric columns...")
    numeric_cols = ["Price", "Qty", "Owned", "Value"]
    for col in numeric_cols:
        if col in df.columns:
            before_conversion = df[col].notna().sum()
            df[col] = pd.to_numeric(
                df[col].astype(str).replace({r"\$": "", ",": ""}, regex=True),
                errors="coerce",
            )
            after_conversion = df[col].notna().sum()
            conversion_loss = before_conversion - after_conversion
            # print(f"[MAIN-DEBUG] {col}: {conversion_loss} values became NaN during conversion")

    # dOwn conversion
    if "dOwn" in df.columns:
        before_down = df["dOwn"].notna().sum()
        df["dOwn"] = pd.to_numeric(
            df["dOwn"].astype(str).replace({r"%": "", r"\+": "", r"New": "999", r">": ""}, regex=True),
            errors="coerce",
        )
        after_down = df["dOwn"].notna().sum()
        # print(f"[MAIN-DEBUG] dOwn: {before_down - after_down} values became NaN during conversion")
    else:
        df["dOwn"] = np.nan
        # print("[MAIN-DEBUG] No dOwn column found, filled with NaN")

    # Drop rows with missing critical values
    before_value_drop = len(df)
    df.dropna(subset=["Filing Date", "Trade Date", "Value"], inplace=True)
    value_dropped = before_value_drop - len(df)
    # print(f"[MAIN-DEBUG] Dropped {value_dropped} rows due to missing Filing Date, Trade Date, or Value")

    # Step 4: Filtering
    if "Trade Type" in df.columns:
        before_purchase_filter = len(df)
        df = df[df["Trade Type"].str.contains("P - Purchase", na=False)].copy()
        purchase_filtered = before_purchase_filter - len(df)
        # print(f"[MAIN-DEBUG] Filtered out {purchase_filtered} non-purchase transactions")
    else:
        print("[MAIN-WARN] No Trade Type column found, assuming all are purchases")

    if df.empty:
        print("[MAIN-ERROR] No purchase transactions found after filtering.")
        return pd.DataFrame()

    print(f"[MAIN-INFO] After filtering: {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 5: Feature Engineering
    print("[MAIN-INFO] Adding derived features...")
    df["Days_Since_Trade"] = (df["Filing Date"] - df["Trade Date"]).dt.days
    # print(f"[MAIN-DEBUG] Days_Since_Trade - Mean: {df['Days_Since_Trade'].mean():.1f}, Max: {df['Days_Since_Trade'].max()}")
    
    df = _parse_insider_titles(df)

    # Step 6: Aggregation
    print("[MAIN-INFO] Aggregating daily trades...")
    df_agg = _aggregate_daily_trades(df)
    
    # Step 7: Final feature engineering
    print("[MAIN-INFO] Adding final date features...")
    df_final = add_date_features(df_agg)
    
    # Add log transformations
    if "Value" in df_final.columns:
        df_final["log_Value"] = np.log1p(df_final["Value"].fillna(0))
    if "Qty" in df_final.columns:
        df_final["log_Qty"] = np.log1p(df_final["Qty"].fillna(0))
    if "Price" in df_final.columns:
        df_final["log_Price"] = np.log1p(df_final["Price"].fillna(0))
    
    # print(f"[MAIN-DEBUG] Added log transformations for Value, Qty, Price")

    # Final column selection
    final_cols = [
        "Ticker", "Filing Date", "Number_of_Purchases", "Price", "Qty", "Owned", "dOwn", "Value",
        "Days_Since_Trade", "CEO", "CFO", "Pres", "VP", "Dir", "TenPercent",
        "log_Value", "log_Qty", "log_Price", "Day_Of_Year", "Day_Of_Quarter"
    ]
    
    available_cols = [col for col in final_cols if col in df_final.columns]
    missing_cols = [col for col in final_cols if col not in df_final.columns]
    
    # print(f"[MAIN-DEBUG] Available final columns: {len(available_cols)}")
    if missing_cols:
        print(f"[MAIN-WARN] Missing expected columns: {missing_cols}")
    
    df_final = df_final.reindex(columns=available_cols)
    
    print(f"\n[MAIN-SUCCESS] Final insider trading dataset: {df_final.shape[0]} rows, {df_final.shape[1]} columns")
    
    return df_final
