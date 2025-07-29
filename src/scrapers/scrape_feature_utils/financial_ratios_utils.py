import pandas as pd
import yfinance as yf
import json
from functools import lru_cache
from pathlib import Path

# --- Constants & Helpers ---
SECTORS = ['Consumer Cyclical', 'Financial Services', 'Healthcare', 'Industrials', 'Technology']

def _get_sector_dummies(sector: str) -> dict:
    """Creates one-hot encoded sector features."""
    return {f'Sector_{s}': 1 if s == sector else 0 for s in SECTORS}

@lru_cache(maxsize=1)
def _load_ticker_to_cik_map(edgar_root: Path) -> dict:
    """Loads the SEC's ticker-to-CIK mapping file once and caches it."""
    try:
        with open(edgar_root / "company_tickers.json") as f:
            company_data = json.load(f)
        return {v['ticker']: str(v['cik_str']) for k, v in company_data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        print("Warning: company_tickers.json not found or is invalid. EDGAR lookup will fail.")
        return {}

def _get_gaap_value(facts: dict, concept: str, unit: str = "USD") -> float | None:
    """Safely extracts the most recent value for a given concept from SEC facts."""
    if concept in facts:
        for item in facts[concept]['units'][unit]:
            if 'val' in item:
                return item['val']
    return None

# --- Data Source Functions ---

def get_ratios_from_edgar(ticker: str, filing_date: pd.Timestamp, edgar_root: Path) -> dict:
    """
    Efficiently finds the latest 10-K or 10-Q before a filing date and extracts key ratios.
    """
    ticker_to_cik = _load_ticker_to_cik_map(edgar_root)
    cik = ticker_to_cik.get(ticker.upper())
    if not cik: return {}
    cik_path = edgar_root / cik
    if not cik_path.is_dir(): return {}

    latest_filing_path = None
    latest_filing_date = pd.Timestamp.min.tz_localize('UTC')

    for form_type in ["10-K", "10-Q"]:
        form_path = cik_path / form_type
        if not form_path.is_dir(): continue
        for submission_file in form_path.glob("*.json"):
            with open(submission_file) as f:
                try:
                    meta = json.load(f)
                    current_filed_date = pd.to_datetime(meta.get('filed'), utc=True)
                    if latest_filing_date < current_filed_date <= filing_date:
                        latest_filing_date = current_filed_date; latest_filing_path = submission_file
                except (json.JSONDecodeError, TypeError): continue

    if not latest_filing_path: return {}
    try:
        with open(latest_filing_path) as f:
            data = json.load(f)
        facts = data.get('facts', {}).get('us-gaap', {})
        revenue = _get_gaap_value(facts, 'Revenues')
        net_income = _get_gaap_value(facts, 'NetIncomeLoss')
        ratios = {'Net_Profit_Margin': net_income / revenue if net_income and revenue else None}
        return {k: v for k, v in ratios.items() if v is not None}
    except Exception:
        return {}

@lru_cache(maxsize=512)
def get_ratios_from_yf(ticker: str) -> dict:
    """Retrieve ratios from yfinance. Cached on ticker as fundamentals don't change daily."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials.iloc[:, 0] if not stock.financials.empty else pd.Series()
        cashflow = stock.cashflow.iloc[:, 0] if not stock.cashflow.empty else pd.Series()
        market_cap = info.get('marketCap')
        net_income = financials.get('Net Income')
        ratios = {
            'Market_Cap': market_cap, 'Price_to_Earnings_Ratio': info.get('trailingPE'),
            'Price_to_Book_Ratio': info.get('priceToBook'), 'Price_to_Sales_Ratio': info.get('priceToSalesTrailing12Months'),
            'Beta_y': info.get('beta'), 'EPS': info.get('trailingEps'),
            'Net_Profit_Margin': net_income / financials.get('Total Revenue') if net_income and financials.get('Total Revenue') else None,
            'Investing_Cash_Flow': cashflow.get('Investing Cash Flow'), 'Financing_Cash_Flow': cashflow.get('Financing Cash Flow'),
            'Free_Cash_Flow': cashflow.get('Free Cash Flow'),
            'Operating_Cash_Flow_to_Market_Cap': cashflow.get('Operating Cash Flow') / market_cap if market_cap else None,
            'Net_Income_to_Market_Cap': net_income / market_cap if market_cap and net_income else None
        }
        sector = info.get('sector', 'Other')
        ratios.update(_get_sector_dummies(sector))
        return {k: v for k, v in ratios.items() if v is not None}
    except Exception:
        return {}

# --- Main Orchestrator Function ---

# **FIX: Function signature updated to accept the edgar_data_path directly**
def get_financial_ratios_for_filing(ticker: str, filing_date: pd.Timestamp, edgar_data_path: Path) -> dict:
    """
    Main entry point. Tries EDGAR first for robust data, then falls back to yfinance.
    """
    # 1. Get fundamental data from EDGAR
    edgar_ratios = get_ratios_from_edgar(ticker, filing_date, edgar_data_path)
    
    # 2. Get data from yfinance (good for market data and as a fallback)
    yf_ratios = get_ratios_from_yf(ticker)
    
    # 3. Combine them: yfinance data acts as the base, EDGAR data overwrites common fields
    combined_ratios = yf_ratios.copy()
    combined_ratios.update(edgar_ratios)
    
    return combined_ratios
