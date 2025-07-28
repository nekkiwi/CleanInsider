from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .. import config


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


def scrape_openinsider(num_weeks: int, output_path: Path) -> pd.DataFrame:
    """
    Scrapes, processes, and engineers features from OpenInsider data.
    """
    # (The rest of this function is unchanged)
    print(f"Scraping latest {num_weeks} weeks of filings...")
    url = f"http://openinsider.com/latest-insider-filings-stocks?weeks={num_weeks}"
    try:
        response = requests.get(url, headers=config.REQUESTS_HEADER)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching data from OpenInsider: {e}")
        return pd.DataFrame(), output_path

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'tinytable'})
    if not table:
        print("Could not find the data table on OpenInsider.")
        return pd.DataFrame(), output_path

    headers = [header.text.strip() for header in table.find_all('th')]
    rows = table.find_all('tr')[1:]
    data = [[ele.text.strip() for ele in row.find_all('td')] for row in tqdm(rows, desc="Parsing OpenInsider rows") if len(row.find_all('td')) == len(headers)]

    df = pd.DataFrame(data, columns=headers)

    df.rename(columns={
        'Filing\xa0Date': 'Filing Date', 'Trade\xa0Date': 'Trade Date',
        'Trade\xa0Type': 'Trade Type', 'ΔOwn': 'dOwn'
    }, inplace=True, errors='ignore')
    df.dropna(subset=['Ticker'], inplace=True)

    df['Filing Date'] = pd.to_datetime(df['Filing Date'], errors='coerce')
    df['Trade Date'] = pd.to_datetime(df['Trade Date'], errors='coerce')
    numeric_cols = ['Price', 'Qty', 'Owned', 'Value']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].replace({r'\$': '', ',': ''}, regex=True),
                errors='coerce',
            )
    if 'dOwn' in df.columns:
        df['dOwn'] = df['dOwn'].str.replace('%', '', regex=False).astype(
            float, errors='ignore'
        )
    else:
        df['dOwn'] = np.nan
    df.dropna(subset=['Filing Date', 'Trade Date', 'Value'], inplace=True)

    if 'Trade Type' in df.columns:
        df = df[df['Trade Type'].str.contains('P - Purchase', na=False)].copy()
    if df.empty:
        print("No purchase transactions found in the specified period.")
        return pd.DataFrame(), output_path

    df['Days_Since_Trade'] = (df['Filing Date'] - df['Trade Date']).dt.days
    df = _parse_insider_titles(df)

    print("Aggregating daily trades and engineering features...")
    df_final = _aggregate_daily_trades(df)

    final_cols = [
        'Ticker', 'Filing Date', 'Number_of_Purchases', 'Price', 'Qty', 'Owned',
        'dOwn', 'Value', 'Days_Since_Trade', 'CEO', 'CFO', 'Pres', 'VP',
        'Dir', 'TenPercent'
    ]
    df_final = df_final.reindex(columns=final_cols)

    df_final.to_excel(output_path, index=False)
    print(f"Stage 1 data saved to {output_path}")

    return df_final, output_path
