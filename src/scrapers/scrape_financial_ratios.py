import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json
import requests

# **FIX: Import the correct utility function**
from .scrape_feature_utils.financial_ratios_utils import get_financial_ratios_for_filing

def _ensure_sec_mapping_file_exists(config):
    """
    Checks for the SEC's ticker-to-CIK mapping file and downloads it if missing.
    This is an automated setup step to ensure the EDGAR scraper can function.
    """
    edgar_path = config.EDGAR_DATA_PATH
    mapping_file_path = edgar_path / "company_tickers.json"

    if mapping_file_path.exists():
        # File is already there, no action needed.
        return

    print("SEC mapping file not found. Attempting to download...")
    
    # The official SEC URL for the ticker-to-CIK mapping file.
    sec_url = "https://www.sec.gov/files/company_tickers.json"
    
    try:
        response = requests.get(sec_url, headers=config.REQUESTS_HEADER)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        data = response.json()
        
        # Ensure the directory exists before writing the file
        edgar_path.mkdir(parents=True, exist_ok=True)
        
        with open(mapping_file_path, 'w') as f:
            json.dump(data, f)
            
        print(f"Successfully downloaded and saved SEC mapping file to: {mapping_file_path}")
    except requests.RequestException as e:
        print(f"Warning: Failed to download SEC mapping file: {e}")
        print("Continuing without EDGAR data. Financial ratios will rely solely on yfinance.")
    except Exception as e:
        print(f"An unexpected error occurred while downloading the SEC mapping file: {e}")

# **FIX: Function signature updated to accept explicit paths**
def scrape_all_financial_ratios(base_df: pd.DataFrame, output_path: Path, edgar_data_path: Path):
    """
    Orchestrates scraping of all financial ratio and sector features,
    using the new EDGAR-first, yfinance-fallback logic.
    """
    filings_to_process = base_df[['Ticker', 'Filing Date']].drop_duplicates()
    all_features = []
    
    print(f"Processing {len(filings_to_process)} unique Ticker/Date pairs for financial features...")
    for _, row in tqdm(filings_to_process.iterrows(), total=len(filings_to_process), desc="Fetching Financial Features"):
        
        # **FIX: Call the utility function with the edgar_data_path**
        features = get_financial_ratios_for_filing(row['Ticker'], row['Filing Date'], edgar_data_path)
        
        # Add identifying keys back to the feature dictionary
        features['Ticker'] = row['Ticker']
        features['Filing Date'] = row['Filing Date']
        all_features.append(features)
        
    if not all_features:
        features_df = pd.DataFrame(columns=['Ticker', 'Filing Date'])
    else:
        features_df = pd.DataFrame(all_features)
        cols = features_df.columns.tolist()
        if 'Ticker' in cols and 'Filing Date' in cols:
            cols.insert(0, cols.pop(cols.index('Filing Date')))
            cols.insert(0, cols.pop(cols.index('Ticker')))
            features_df = features_df.reindex(columns=cols)

    # **FIX: Use the output_path argument directly**
    features_df.to_excel(output_path, index=False)
    print(f"Stage 3 data saved to {output_path}")

