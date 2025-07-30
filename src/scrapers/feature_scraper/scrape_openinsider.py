# file: /src/scrapers/scrape_openinsider.py (Joblib + TQDM Version)

import datetime
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Joblib is more stable for these kinds of data science workloads
from joblib import Parallel, delayed
from tqdm import tqdm

# Assuming you have a config.py file with your header
import config

from .feature_scraper_util.general_utils import add_date_features

# --- HELPER FUNCTIONS (UNCHANGED) ---


def _parse_insider_titles(df: pd.DataFrame) -> pd.DataFrame:
    # This function is correct and remains unchanged
    if "Title" not in df.columns:
        df["CEO"] = 0
        df["CFO"] = 0
        df["Pres"] = 0
        df["VP"] = 0
        df["Dir"] = 0
        df["TenPercent"] = 0
        return df
    df["Title_lower"] = df["Title"].str.lower()
    df["CEO"] = (
        df["Title_lower"]
        .str.contains("ceo|chief executive officer", na=False)
        .astype(int)
    )
    df["CFO"] = (
        df["Title_lower"]
        .str.contains("cfo|chief financial officer", na=False)
        .astype(int)
    )
    df["Pres"] = df["Title_lower"].str.contains("pres|president", na=False).astype(int)
    df["VP"] = df["Title_lower"].str.contains("vp|vice president", na=False).astype(int)
    df["Dir"] = df["Title_lower"].str.contains("dir|director", na=False).astype(int)
    df["TenPercent"] = (
        df["Title_lower"].str.contains("10%|ten percent", na=False).astype(int)
    )
    df.drop(columns=["Title_lower", "Title"], inplace=True)
    return df


def _aggregate_daily_trades(df: pd.DataFrame) -> pd.DataFrame:
    # This function is correct and remains unchanged
    agg_funcs = {
        "Value": "sum",
        "Qty": "sum",
        "Owned": "last",
        "dOwn": "last",
        "CEO": "max",
        "CFO": "max",
        "Pres": "max",
        "VP": "max",
        "Dir": "max",
        "TenPercent": "max",
        "Days_Since_Trade": "mean",
    }
    grouped = df.groupby(["Ticker", pd.Grouper(key="Filing Date", freq="D")])
    df_agg = grouped.agg(agg_funcs)
    if "Price" in df.columns and df["Price"].notna().any():
        df_agg["Price"] = grouped.apply(
            lambda x: np.average(x["Price"], weights=x["Value"]), include_groups=False
        )
    else:
        df_agg["Price"] = np.nan
    df_agg["Number_of_Purchases"] = grouped.size()
    return df_agg.reset_index()


# --- WORKER FUNCTION (NOW MORE SELF-CONTAINED) ---


def _scrape_date_range_worker(
    date_range: tuple, request_header: dict
) -> pd.DataFrame | None:
    """
    Worker function that scrapes all pages for a given date range.
    Now accepts the request header as an argument for better parallel safety.
    """
    start_date, end_date = date_range
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
                break
            df = pd.read_html(StringIO(str(table)))[0]
            if df.empty:
                break
            all_data_for_range.append(df)
            if len(df) < 1000:
                break
            page += 1
            time.sleep(0.25)
        except requests.RequestException:
            break

    if not all_data_for_range:
        return None

    full_df = pd.concat(all_data_for_range, ignore_index=True)
    full_df.columns = full_df.columns.str.replace("\xa0", " ", regex=False)
    return full_df


# --- MAIN ORCHESTRATOR (REFACTORED TO USE JOBLIB) ---


def scrape_openinsider(num_weeks: int) -> pd.DataFrame:
    """
    Scrapes, processes, and engineers features from OpenInsider using a robust,
    parallelized approach with joblib and tqdm.
    """
    # Step 1: Create date ranges for scraping
    end_date = datetime.datetime.now()
    date_ranges = []
    for _ in range(num_weeks):
        start_date = end_date - datetime.timedelta(days=7)
        date_ranges.append((start_date, end_date))
        end_date = start_date - datetime.timedelta(days=1)

    # Step 2: Use joblib to fetch data in parallel
    print(f"Initiating parallel scrape for {num_weeks} weeks of data...")
    # Use n_jobs=-2 to use all CPU cores but one, keeping the system responsive.
    n_jobs = -2

    # Create a list of delayed function calls
    tasks = [
        delayed(_scrape_date_range_worker)(dr, config.REQUESTS_HEADER)
        for dr in date_ranges
    ]

    # Run the tasks in parallel, wrapped with tqdm for a progress bar
    results = Parallel(n_jobs=n_jobs)(tqdm(tasks, desc="Scraping weekly data"))

    # Filter out None results from failed scrapes
    all_data_frames = [df for df in results if df is not None]
    if not all_data_frames:
        print("No data was scraped. Exiting.")
        return pd.DataFrame()

    df = pd.concat(all_data_frames, ignore_index=True).drop_duplicates()

    # Step 3: Data Cleaning and Type Conversion
    df.rename(columns={"ΔOwn": "dOwn"}, inplace=True, errors="ignore")
    df.dropna(subset=["Ticker"], inplace=True)
    df["Filing Date"] = pd.to_datetime(df["Filing Date"], errors="coerce")
    df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")

    numeric_cols = ["Price", "Qty", "Owned", "Value"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).replace({r"\$": "", ",": ""}, regex=True),
                errors="coerce",
            )
    if "dOwn" in df.columns:
        df["dOwn"] = pd.to_numeric(
            df["dOwn"]
            .astype(str)
            .replace({r"%": "", r"\+": "", r"New": "999", r">": ""}, regex=True),
            errors="coerce",
        )
    else:
        df["dOwn"] = np.nan
    df.dropna(subset=["Filing Date", "Trade Date", "Value"], inplace=True)

    # Step 4: Filtering and Feature Engineering
    if "Trade Type" in df.columns:
        df = df[df["Trade Type"].str.contains("P - Purchase", na=False)].copy()
    if df.empty:
        print("No purchase transactions found.")
        return pd.DataFrame()

    df["Days_Since_Trade"] = (df["Filing Date"] - df["Trade Date"]).dt.days
    df = _parse_insider_titles(df)

    # Step 5: Aggregation and Final Touches
    print("Aggregating daily trades and engineering features...")
    df_agg = _aggregate_daily_trades(df)
    df_final = add_date_features(df_agg)

    final_cols = [
        "Ticker",
        "Filing Date",
        "Number_of_Purchases",
        "Price",
        "Qty",
        "Owned",
        "dOwn",
        "Value",
        "Days_Since_Trade",
        "CEO",
        "CFO",
        "Pres",
        "VP",
        "Dir",
        "TenPercent",
        "Day_Of_Year",
        "Day_Of_Quarter",
    ]
    # Use reindex to ensure all columns exist, filling missing ones with NaN
    df_final = df_final.reindex(columns=final_cols)

    return df_final


# --- STANDALONE TEST BLOCK ---
if __name__ == "__main__":
    print("Running scrape_openinsider.py in standalone test mode...")
    # This requires a config.py file in the same directory with:
    # REQUESTS_HEADER = {'User-Agent': 'your.name@yourdomain.com'}

    final_df = scrape_openinsider(num_weeks=4)

    if not final_df.empty:
        print("\nStandalone test complete. Sample output:")
        print(final_df.head())
        final_df.to_csv("openinsider_summary_standalone.csv", index=False)
