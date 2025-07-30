# file: /src/scrapers/load_sec_data.py (Joblib + TQDM Version)

import requests
import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import Counter

# Import the necessary libraries for robust parallelism
from joblib import Parallel, delayed
from tqdm import tqdm

# --- HELPER FUNCTIONS (Self-Contained and Unchanged) ---
def _pre_filter_sec_columns(all_rows: list[dict], presence_threshold: float = 0.01) -> list[dict]:
    """
    Analyzes all feature keys before creating a DataFrame to remove extremely rare ones.
    This prevents the DataFrame from becoming excessively wide and causing memory errors.

    Args:
        all_rows (list[dict]): The list of dictionaries, where each is a future row.
        presence_threshold (float): A feature must be present in at least this fraction
                                    of rows to be kept. Default is 1%.

    Returns:
        list[dict]: A new list of dictionaries with rare keys removed.
    """
    if not all_rows:
        return []

    print(f"\n--- Pre-filtering SEC columns to save memory ---")
    
    # Count how many times each feature key appears across all dictionaries
    key_counter = Counter(key for row_dict in all_rows for key in row_dict)
    
    # The minimum number of times a key must appear to be kept
    min_occurrences = int(len(all_rows) * presence_threshold)
    
    # Identify which keys to keep. Always keep the identifiers.
    keys_to_keep = {'Ticker', 'Filing Date'}
    for key, count in key_counter.items():
        if count >= min_occurrences:
            keys_to_keep.add(key)
    
    original_key_count = len(key_counter)
    print(f"   → Found {original_key_count} unique SEC tags.")
    print(f"   → Keeping {len(keys_to_keep)} features present in at least {presence_threshold:.1%} of rows.")

    # Create a new list of dictionaries, containing only the keys we want to keep
    # This is much more memory-efficient than creating a huge DataFrame and then dropping columns.
    filtered_rows = [
        {key: row_dict.get(key) for key in keys_to_keep if key in row_dict}
        for row_dict in all_rows
    ]
    
    return filtered_rows

def fetch_ticker_cik_map(request_header: str):
    """Fetches the ticker-to-CIK mapping from the SEC."""
    print("🔄 Fetching ticker->CIK map from SEC...")
    try:
        resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=request_header)
        resp.raise_for_status()
        data = resp.json()
        mapping = {str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
        print(f"   → Retrieved {len(mapping)} entries.")
        return mapping
    except Exception as e:
        print(f"   ❌ Failed to fetch CIK map: {e}")
        return {}

def add_engineered_features(s: pd.Series) -> pd.Series:
    """Calculates aggregate financial ratios and growth metrics."""
    def _safe_div(num, den):
        return num / den if pd.notna(num) and pd.notna(den) and den != 0 else np.nan
    for q in [1, 2]:
        revenue = s.get(f'RevenueFromContractWithCustomerExcludingAssessedTax_q-{q}')
        gross_profit = s.get(f'GrossProfit_q-{q}')
        s[f'FE_GrossMargin_q{q}'] = _safe_div(gross_profit, revenue)
    s['FE_GrossMargin_Change'] = s.get('FE_GrossMargin_q1') - s.get('FE_GrossMargin_q2')
    return s

# --- NEW: WORKER FUNCTION FOR PARALLEL PRE-LOADING ---

def _process_single_quarter(q_name: str, parquet_dir: Path) -> tuple[str, pd.Series | None]:
    """
    Worker function to process a single quarter's data. This is the heavy lifting.
    Returns a tuple of the quarter name and the processed data Series.
    """
    q_path = parquet_dir / "quarter" / f"{q_name}.zip"
    if not q_path.exists():
        return q_name, None
    try:
        sub = pd.read_parquet(q_path / "sub.txt.parquet", columns=['adsh', 'cik'])
        num = pd.read_parquet(q_path / "num.txt.parquet", columns=['adsh', 'tag', 'ddate', 'value'])
        merged = pd.merge(sub, num, on='adsh')
        merged = merged.sort_values('ddate')
        # Group by CIK and tag, keeping the last reported value for that quarter
        processed = merged.groupby(['cik', 'tag'])['value'].last()
        return q_name, processed
    except Exception as e:
        # Silently fail for a single bad file to not stop the whole pipeline
        return q_name, None

# --- REFACTORED: PRE-COMPUTATION FUNCTION USING JOBLIB ---

def preload_and_process_sec_quarters(parquet_dir: Path, quarters: list) -> dict:
    """
    Uses joblib to load and process all required quarterly SEC data in parallel.
    """
    print(f"--- Starting SEC Data Pre-computation for {len(quarters)} quarters (in parallel) ---")
    
    # Use n_jobs=-2 to use all CPU cores but one, keeping the system responsive.
    n_jobs = -2
    
    # Create a list of delayed function calls for joblib
    tasks = [delayed(_process_single_quarter)(q_name, parquet_dir) for q_name in quarters]
    
    # Run the tasks in parallel with a tqdm progress bar
    results = Parallel(n_jobs=n_jobs)(
        tqdm(tasks, desc="Processing SEC Quarters")
    )

    # Assemble the results into the data warehouse dictionary
    data_warehouse = {q_name: data for q_name, data in results if data is not None}
            
    print("--- ✅ SEC Data Pre-computation Phase Complete ---\n")
    return data_warehouse

# --- MAIN ORCHESTRATOR FOR THIS MODULE ---

def load_sec_features_df(input_df, parquet_dir_str, request_header, n_prev=2):
    """The main entry point function for this module."""
    parquet_root = Path(parquet_dir_str)
    
    # 1. Determine unique quarters needed (this is fast)
    all_quarters_needed = set()
    dates = pd.to_datetime(input_df['Filing Date'])
    for i in range(n_prev):
        periods = (dates.dt.to_period('Q-DEC') - i)
        q_names = {f"{p.year}q{p.quarter}" for p in periods}
        all_quarters_needed.update(q_names)

    # 2. Call the parallel pre-loader
    data_warehouse = preload_and_process_sec_quarters(parquet_root, sorted(list(all_quarters_needed)))
    
    # 3. Perform the fast, sequential lookup phase
    cik_map = fetch_ticker_cik_map(request_header)
    input_df['CIK'] = input_df['Ticker'].str.upper().map(cik_map)
    
    all_rows = []
    print("--- Starting SEC Feature Lookup (Serial) ---")
    for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Looking up SEC Features"):
        ticker, fdate_str, cik = row['Ticker'], row['Filing Date'], row['CIK']
        if pd.isna(cik): continue
        cik = int(cik)
        
        period = pd.Period(fdate_str, freq="Q-DEC")
        quarters_to_check = [f"{(period - i).year}q{(period - i).quarter}" for i in range(n_prev)]
        series_map = {}
        for i, q_name in enumerate(quarters_to_check, start=1):
            quarterly_data = data_warehouse.get(q_name)
            if quarterly_data is None: continue
            try:
                cik_facts = quarterly_data.loc[cik] 
                s = cik_facts.rename(index=lambda tag: f"{tag}_q-{i}")
                series_map[i] = s
            except KeyError: continue
        
        if not series_map: continue

        all_s = pd.concat(series_map.values())
        all_s = all_s[~all_s.index.duplicated()]
        all_base_tags = {t.rsplit("_q-", 1)[0] for t in all_s.index}
        if 1 in series_map:
            for j in range(2, n_prev + 1):
                if j in series_map:
                    for tag in all_base_tags:
                        val1 = all_s.get(f"{tag}_q-1", np.nan)
                        valj = all_s.get(f"{tag}_q-{j}", np.nan)
                        all_s[f"{tag}_diff_q1{j}"] = val1 - valj
        all_s = add_engineered_features(all_s)
        row_dict = {"Ticker": ticker, "Filing Date": fdate_str}
        row_dict.update(all_s.to_dict())
        all_rows.append(row_dict)

    print("--- ✅ SEC Feature Lookup Complete ---\n")
    filtered_rows = _pre_filter_sec_columns(all_rows, presence_threshold=0.2) # Keep features in at least 20% of rows
    
    if not filtered_rows:
        return pd.DataFrame()
        
    return pd.DataFrame(filtered_rows)

# --- STANDALONE TEST BLOCK ---
if __name__ == '__main__':
    print("Running load_sec_data.py in standalone test mode...")
    PARQUET_DIR = "../../data/sec_database/parquet"
    USER_AGENT = "your.name@yourdomain.com"
    data = {"Ticker": ["AAPL", "MSFT", "GOOG"], "Filing Date": ["2024-01-28", "2024-01-24", "2023-10-30"]}
    input_df = pd.DataFrame(data)
    
    sec_features_df = load_sec_features_df(input_df, PARQUET_DIR, USER_AGENT)
    
    if not sec_features_df.empty:
        print("\nStandalone test complete. Sample output:")
        print(sec_features_df.head())
        sec_features_df.to_csv("sec_features_summary_standalone.csv", index=False)
