from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from pathlib import Path
import time
import datetime
from io import StringIO
from multiprocessing import Pool, cpu_count

# --- Third-Party Imports ---
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# insert the project root so that absolute imports will work
import config 
from scrapers.scrape_feature_utils.general_utils import add_date_features


def _parse_insider_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the 'Title' column to create one-hot encoded roles.
    This version is more robust against missing title data.
    """
    if 'Title' not in df.columns:
        df['CEO'] = 0
        df['CFO'] = 0
        df['Pres'] = 0
        df['VP'] = 0
        df['Dir'] = 0
        df['TenPercent'] = 0
        return df

    df['Title_lower'] = df['Title'].str.lower()
    df['CEO'] = df['Title_lower'].str.contains('ceo|chief executive officer', na=False).astype(int)
    df['CFO'] = df['Title_lower'].str.contains('cfo|chief financial officer', na=False).astype(int)
    df['Pres'] = df['Title_lower'].str.contains('pres|president', na=False).astype(int)
    df['VP'] = df['Title_lower'].str.contains('vp|vice president', na=False).astype(int)
    df['Dir'] = df['Title_lower'].str.contains('dir|director', na=False).astype(int)
    df['TenPercent'] = df['Title_lower'].str.contains('10%|ten percent', na=False).astype(int)
    df.drop(columns=['Title_lower', 'Title'], inplace=True)
    return df

def _aggregate_daily_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates all trades for a given Ticker that were filed on the same day.
    """
    # Ensure all expected columns exist so aggregation doesn't fail on missing data
    for col in ['Value', 'Qty', 'Owned', 'dOwn', 'Price']:
        if col not in df.columns:
            df[col] = np.nan

    agg_funcs = {
        'Value': 'sum',
        'Qty': 'sum',
        'Owned': 'last',
        'dOwn': 'last',
        'CEO': 'max',
        'CFO': 'max',
        'Pres': 'max',
        'VP': 'max',
        'Dir': 'max',
        'TenPercent': 'max',
        'Days_Since_Trade': 'mean',
    }
    grouped = df.groupby(['Ticker', pd.Grouper(key='Filing Date', freq='D')])
    df_agg = grouped.agg(agg_funcs)

    # **FIX: Added include_groups=False to silence the FutureWarning**
    if df['Price'].notna().any():
        df_agg['Price'] = grouped.apply(
            lambda x: np.average(x['Price'], weights=x['Value']),
            include_groups=False,
        )
    else:
        df_agg['Price'] = np.nan

    df_agg['Number_of_Purchases'] = grouped.size()
    return df_agg.reset_index()


def _scrape_date_range_worker(date_range: tuple) -> pd.DataFrame:
    """
    Worker function for multiprocessing. Scrapes ALL pages for a given date range.
    This logic is adapted from your FeatureScraper class, with pagination added.
    """
    start_date, end_date = date_range
    base_url = "http://openinsider.com/screener?"
    all_data_for_range = []
    page = 1
    
    date_str = (
        f"{start_date.month}%2F{start_date.day}%2F{start_date.year}+-+"
        f"{end_date.month}%2F{end_date.day}%2F{end_date.year}"
    )

    while True:
        # Build the URL using the screener with date range parameters
        url = (
            f"{base_url}pl=1&ph=&ll=&lh=&fd=-1&fdr={date_str}&td=0&tdr=&fdlyl=&fdlyh="
            f"&daysago=&xp=1&vl=10&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh="
            f"&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page={page}"
        )
        try:
            response = requests.get(url, headers=config.REQUESTS_HEADER)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'tinytable'})

            # Stop if there's no table on the page
            if table is None: break

            # Read table into DataFrame; stop if it's empty
            df = pd.read_html(StringIO(str(table)))[0]
            if df.empty: break

            all_data_for_range.append(df)
            
            # Stop if the last page is reached (less than 1000 results)
            if len(df) < 1000: break

            page += 1
            time.sleep(0.25) # Be polite to the server
        except requests.RequestException:
            break # Stop if a page request fails

    if not all_data_for_range:
        return None
        
    # Clean up column names right away to ensure consistency
    full_df = pd.concat(all_data_for_range, ignore_index=True)
    full_df.columns = full_df.columns.str.replace('\xa0', ' ', regex=False)
    return full_df


# ##################################################################
# ## SECTION 2: REPLACEMENT `scrape_openinsider` FUNCTION         ##
# ##################################################################

def scrape_openinsider(num_weeks: int, output_path: Path = None) -> pd.DataFrame:
    """
    Scrapes, processes, and engineers features from OpenInsider purchase data
    using a parallelized, paginated approach based on the screener URL.
    
    Args:
        num_weeks (int): The number of recent weeks of filings to scrape.
        output_path (Path, optional): If provided, the final DataFrame will be
                                      saved as an Excel file to this path. 
                                      Defaults to None.

    Returns:
        pd.DataFrame: A fully cleaned, processed, and feature-engineered DataFrame.
    """
    
    # --- Step 1: Parallel & Paginated Data Ingestion ---
    print(f"Initiating parallel scrape for {num_weeks} weeks of data...")
    
    # Create weekly date ranges to scrape, starting from 30 days ago
    end_date = datetime.datetime.now() - datetime.timedelta(days=30)
    date_ranges = []
    for _ in range(num_weeks):
        start_date = end_date - datetime.timedelta(days=7)
        date_ranges.append((start_date, end_date))
        end_date = start_date

    # Use multiprocessing to fetch data for each date range in parallel
    num_processes = min(cpu_count(), len(date_ranges))
    all_data_frames = []
    with Pool(num_processes) as pool:
        # Use tqdm to create a progress bar for the scraping process
        with tqdm(total=len(date_ranges), desc="- Scraping weekly data from OpenInsider") as pbar:
            for df_chunk in pool.imap_unordered(_scrape_date_range_worker, date_ranges):
                if df_chunk is not None:
                    all_data_frames.append(df_chunk)
                pbar.update(1)

    if not all_data_frames:
        print("No data was scraped. Exiting.")
        return pd.DataFrame()

    df = pd.concat(all_data_frames, ignore_index=True).drop_duplicates()

    # --- Step 2: Data Cleaning and Type Conversion ---
    df.rename(columns={'ΔOwn': 'dOwn'}, inplace=True, errors='ignore')
    df.dropna(subset=['Ticker'], inplace=True)
    df['Filing Date'] = pd.to_datetime(df['Filing Date'], errors='coerce')
    df['Trade Date'] = pd.to_datetime(df['Trade Date'], errors='coerce')
    
    numeric_cols = ['Price', 'Qty', 'Owned', 'Value']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace({r'\$': '', ',': ''}, regex=True), errors='coerce')
    if 'dOwn' in df.columns:
        df['dOwn'] = pd.to_numeric(df['dOwn'].replace({r'%': '', r'\+': '', r'New': '999', r'>': ''}, regex=True), errors='coerce')
    else:
        df['dOwn'] = np.nan
    df.dropna(subset=['Filing Date', 'Trade Date', 'Value'], inplace=True)

    # --- Step 3: Filtering and Feature Engineering ---
    # The URL param 'xp=1' already filters for purchases, but this is a good safeguard.
    if 'Trade Type' in df.columns:
        df = df[df['Trade Type'].str.contains('P - Purchase', na=False)].copy()
        
    if df.empty:
        print("No purchase transactions were found in the specified period.")
        return pd.DataFrame()

    df['Days_Since_Trade'] = (df['Filing Date'] - df['Trade Date']).dt.days
    df = _parse_insider_titles(df)

    # --- Step 4: Aggregation and Final Touches ---
    print("Aggregating daily trades and engineering features...")
    df_agg = _aggregate_daily_trades(df)
    df_final = add_date_features(df_agg)

    final_cols = [
        'Ticker', 'Filing Date', 'Number_of_Purchases', 'Price', 'Qty', 'Owned',
        'dOwn', 'Value', 'Days_Since_Trade', 'CEO', 'CFO', 'Pres', 'VP',
        'Dir', 'TenPercent', 'Day_Of_Year', 'Day_Of_Quarter'
    ]
    df_final = df_final.reindex(columns=final_cols)

    # --- Step 5: Output and Return ---
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_final.to_excel(output_path, index=False)
            print(f"\nSuccess! Stage 1 data saved to {output_path}")
        except Exception as e:
            print(f"\nError saving data to Excel file: {e}")

    return df_final