# file: analyze_filings_fast.py (Corrected)

import requests
import pandas as pd
import numpy as np
from pathlib import Path

# --- Helper function to get CIKs (remains the same) ---
def fetch_ticker_cik_map(user_agent: str):
    print("🔄 Fetching ticker->CIK map from SEC...")
    headers = {"User-Agent": user_agent}
    resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    resp.raise_for_status()
    data = resp.json()
    mapping = {str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
    print(f"   → Retrieved {len(mapping)} entries\n")
    return mapping

# --- Pre-computation function for SEC data ---
def preload_and_process_sec_quarters(parquet_dir: Path, quarters: list) -> dict:
    print("--- Starting SEC Data Pre-computation Phase ---")
    data_warehouse = {}
    for q_name in sorted(list(quarters)):
        print(f"   ▶️  Pre-loading and processing quarter {q_name}...")
        q_path = parquet_dir / "quarter" / f"{q_name}.zip"
        if not q_path.exists():
            print(f"   ❌ Quarter file not found: {q_name}.zip. It will be skipped.")
            continue
        try:
            sub = pd.read_parquet(q_path / "sub.txt.parquet", columns=['adsh', 'cik'])
            num = pd.read_parquet(q_path / "num.txt.parquet", columns=['adsh', 'tag', 'ddate', 'value'])
            
            merged = pd.merge(sub, num, on='adsh')
            merged = merged.sort_values('ddate')
            processed = merged.groupby(['cik', 'tag'])['value'].last()
            
            data_warehouse[q_name] = processed
        except Exception as e:
            print(f"   ❌ Error processing quarter {q_name}: {e}")
            continue
            
    print("--- ✅ SEC Data Pre-computation Phase Complete ---\n")
    return data_warehouse

# --- Feature Engineering function ---
def add_engineered_features(s: pd.Series) -> pd.Series:
    def _safe_div(num, den):
        return num / den if pd.notna(num) and pd.notna(den) and den != 0 else np.nan
    for q in [1, 2]:
        revenue = s.get(f'RevenueFromContractWithCustomerExcludingAssessedTax_q-{q}')
        gross_profit = s.get(f'GrossProfit_q-{q}')
        op_income = s.get(f'OperatingIncomeLoss_q-{q}')
        net_income = s.get(f'NetIncomeLoss_q-{q}')
        s[f'FE_GrossMargin_q{q}'] = _safe_div(gross_profit, revenue)
        s[f'FE_OperatingMargin_q{q}'] = _safe_div(op_income, revenue)
        s[f'FE_NetMargin_q{q}'] = _safe_div(net_income, revenue)
    s['FE_GrossMargin_Change'] = s.get('FE_GrossMargin_q1') - s.get('FE_GrossMargin_q2')
    return s

# --- Main orchestrator for batch processing ---
def load_sec_features_df(input_df, parquet_dir_str, user_agent, n_prev=2):
    parquet_root = Path(parquet_dir_str)
    
    print("--- Determining all unique quarters required for the batch run ---")
    all_quarters_needed = set()
    dates = pd.to_datetime(input_df['Filing Date'])
    for i in range(n_prev):
        periods = (dates.dt.to_period('Q-DEC') - 1 - i) 
        q_names = {f"{p.year}q{p.quarter}" for p in periods}
        all_quarters_needed.update(q_names)
    print(f"   → Found {len(all_quarters_needed)} unique quarters to load.")

    data_warehouse = preload_and_process_sec_quarters(parquet_root, all_quarters_needed)
    cik_map = fetch_ticker_cik_map(user_agent)
    input_df['CIK'] = input_df['Ticker'].str.upper().map(cik_map)
    
    print("--- Starting Fast Lookup Phase ---")
    all_rows = []
    for index, row in input_df.iterrows():
        ticker, fdate_str, cik = row['Ticker'], row['Filing Date'], row['CIK']
        
        if pd.isna(cik): continue
        cik = int(cik)
        
        period = (pd.Period(fdate_str, freq="Q-DEC") - 1)
        quarters_to_check = [f"{(period - i).year}q{(period - i).quarter}" for i in range(n_prev)]

        series_map = {}
        for i, q_name in enumerate(quarters_to_check, start=1):
            quarterly_data = data_warehouse.get(q_name)
            if quarterly_data is None: continue
            
            try:
                cik_facts = quarterly_data.loc[cik] 
                s = cik_facts.rename(index=lambda tag: f"{tag}_q-{i}")
                series_map[i] = s
            except KeyError:
                continue
        
        if not series_map: continue

        all_s = pd.concat(series_map.values())
        all_s = all_s[~all_s.index.duplicated()]
        all_base_tags = {t.rsplit("_q-", 1)[0] for t in all_s.index}
        
        if 1 in series_map:
            for j in range(2, n_prev + 1):
                if j in series_map:
                    for tag in all_base_tags:
                        # --- THIS IS THE FIX ---
                        # Use .get(key, np.nan) to handle missing values gracefully.
                        val1 = all_s.get(f"{tag}_q-1", np.nan)
                        valj = all_s.get(f"{tag}_q-{j}", np.nan)
                        all_s[f"{tag}_diff_q1{j}"] = val1 - valj
                        # --- END OF FIX ---
        
        all_s = add_engineered_features(all_s)
        row_dict = {"Ticker": ticker, "Filing Date": fdate_str}
        row_dict.update(all_s.to_dict())
        all_rows.append(row_dict)

    print("--- ✅ Fast Lookup Phase Complete ---\n")
    return pd.DataFrame(all_rows)

if __name__ == '__main__':
    PARQUET_DIR = "../../data/sec_database/parquet"
    USER_AGENT = "your.name@yourdomain.com"
    N_PREVIOUS_QUARTERS = 2
    OUTPUT_CSV = "sec_features_summary_fast.csv"

    data = {
        "Ticker": ["AAPL", "MSFT", "GOOG", "AAPL"],
        "Filing Date": ["2024-01-28", "2024-01-24", "2023-10-30", "2023-10-20"]
    }
    input_df = pd.DataFrame(data)
    print("Input Data:")
    print(input_df)
    
    sec_features_df = load_sec_features_df(input_df, PARQUET_DIR, USER_AGENT, n_prev=N_PREVIOUS_QUARTERS)

    if not sec_features_df.empty:
        sec_features_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 Summary successfully saved to '{OUTPUT_CSV}'")
        print(f"Generated DataFrame with {sec_features_df.shape[0]} rows and {sec_features_df.shape[1]} columns.")
    else:
        print("\nNo data was generated, so no file was saved.")
