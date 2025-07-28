import pandas as pd
import yfinance as yf
from functools import lru_cache

# Define sectors for one-hot encoding
SECTORS = ['Consumer Cyclical', 'Financial Services', 'Healthcare', 'Industrials', 'Technology']

def _get_sector_dummies(sector: str) -> dict:
    """Creates one-hot encoded sector features."""
    return {f'Sector_{s}': 1 if s == sector else 0 for s in SECTORS}

# **FIX: Removed the unhashable 'config' argument from the cached function.**
@lru_cache(maxsize=512)
def get_financial_ratios_for_filing(ticker: str) -> dict:
    """
    Gets a comprehensive set of financial ratios and sector info from yfinance.
    Now cached to avoid repeated API calls for the same ticker.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials.iloc[:, 0] if not stock.financials.empty else pd.Series()
        cashflow = stock.cashflow.iloc[:, 0] if not stock.cashflow.empty else pd.Series()
        
        market_cap = info.get('marketCap')
        net_income = financials.get('Net Income')

        ratios = {
            'Market_Cap': market_cap,
            'Price_to_Earnings_Ratio': info.get('trailingPE'),
            'Price_to_Book_Ratio': info.get('priceToBook'),
            'Price_to_Sales_Ratio': info.get('priceToSalesTrailing12Months'),
            'Beta_y': info.get('beta'),
            'EPS': info.get('trailingEps'),
            'Net_Profit_Margin': financials.get('Net Income') / financials.get('Total Revenue') if financials.get('Total Revenue') else None,
            'Investing_Cash_Flow': cashflow.get('Investing Cash Flow'),
            'Financing_Cash_Flow': cashflow.get('Financing Cash Flow'),
            'Free_Cash_Flow': cashflow.get('Free Cash Flow'),
            'Operating_Cash_Flow_to_Market_Cap': cashflow.get('Operating Cash Flow') / market_cap if market_cap else None,
            'Net_Income_to_Market_Cap': net_income / market_cap if market_cap and net_income else None
        }
        
        sector = info.get('sector', 'Other')
        ratios.update(_get_sector_dummies(sector))

        return {k: v for k, v in ratios.items() if v is not None}
    except Exception:
        return {}
