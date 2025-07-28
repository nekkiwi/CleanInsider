from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .scrape_feature_utils.financial_ratios_utils import (
    get_financial_ratios_for_filing,
)


def scrape_all_financial_ratios(
    base_df: pd.DataFrame, output_path: Path, edgar_root: Path
) -> pd.DataFrame:
    """Orchestrates scraping of all financial ratio and sector features."""
    filings_to_process = base_df[['Ticker', 'Filing Date']].drop_duplicates()
    all_features = []

    print(f"Processing {len(filings_to_process)} unique Ticker/Date pairs for financial features...")
    for _, row in tqdm(filings_to_process.iterrows(), total=len(filings_to_process), desc="Fetching Financial Features"):
        features = get_financial_ratios_for_filing(
            row['Ticker'], row['Filing Date'], edgar_root
        )
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

    features_df.to_excel(output_path, index=False)
    print(f"Stage 3 data saved to {output_path}")

    return features_df
