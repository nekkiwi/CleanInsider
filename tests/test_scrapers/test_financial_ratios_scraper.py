from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from src.scrapers.scrape_financial_ratios import (
    get_financial_ratios_for_filing,
    scrape_all_financial_ratios,
)


def test_get_financial_ratios_for_filing_edgar_success(monkeypatch):
    """
    Tests the success path where the primary method (EDGAR) returns valid features.
    """
    # 1. Arrange: Mock the utility functions
    mock_edgar_data = {'P/E Ratio': 25, 'Debt to Equity': 0.4}
    mock_get_edgar = MagicMock(return_value=mock_edgar_data)
    mock_get_yf = MagicMock(side_effect=AssertionError("Fallback yfinance function should not be called"))

    monkeypatch.setattr('src.scrapers.scrape_feature_utils.financial_ratios_utils.get_ratios_from_edgar', mock_get_edgar)
    monkeypatch.setattr('src.scrapers.scrape_feature_utils.financial_ratios_utils.get_ratios_from_yf', mock_get_yf)

    # 2. Act: Run the function
    features = get_financial_ratios_for_filing('AAPL', pd.to_datetime('2023-01-10'), Path('/fake/edgar'))

    # 3. Assert: Check the results
    mock_get_edgar.assert_called_once()
    assert features['P/E Ratio'] == 25

def test_get_financial_ratios_for_filing_fallback_to_yf(monkeypatch):
    """
    Tests the fallback path where the EDGAR method fails (returns empty)
    and the secondary method (yfinance) succeeds.
    """
    # 1. Arrange: Mock the utility functions
    mock_get_edgar = MagicMock(return_value={}) # Fails by returning empty dict
    mock_yf_data = {'P/E Ratio': 30, 'Current Ratio': 1.5}
    mock_get_yf = MagicMock(return_value=mock_yf_data)

    monkeypatch.setattr('src.scrapers.scrape_feature_utils.financial_ratios_utils.get_ratios_from_edgar', mock_get_edgar)
    monkeypatch.setattr('src.scrapers.scrape_feature_utils.financial_ratios_utils.get_ratios_from_yf', mock_get_yf)

    # 2. Act: Run the function
    features = get_financial_ratios_for_filing('MSFT', pd.to_datetime('2023-03-15'), Path('/fake/edgar'))

    # 3. Assert: Check results and function calls
    mock_get_edgar.assert_called_once()
    mock_get_yf.assert_called_once()
    assert features['P/E Ratio'] == 30

def test_get_financial_ratios_for_filing_all_fail(monkeypatch):
    """
    Tests the failure path where both EDGAR and yfinance methods fail.
    It should return an empty dictionary.
    """
    # 1. Arrange: Mock both utility functions to fail
    mock_get_edgar = MagicMock(return_value={})
    mock_get_yf = MagicMock(return_value={})

    monkeypatch.setattr('src.scrapers.scrape_feature_utils.financial_ratios_utils.get_ratios_from_edgar', mock_get_edgar)
    monkeypatch.setattr('src.scrapers.scrape_feature_utils.financial_ratios_utils.get_ratios_from_yf', mock_get_yf)

    # 2. Act: Run the function
    features = get_financial_ratios_for_filing('TSLA', pd.to_datetime('2023-05-20'), Path('/fake/edgar'))

    # 3. Assert: The result should be an empty dictionary
    assert features == {}

def test_scrape_all_financial_ratios_integration(monkeypatch, tmp_path):
    """
    Tests the main orchestrator function for financial ratios.
    """
    # 1. Arrange: Mock the high-level logic function and prepare data
    mock_feature_dict = {'P/E Ratio': 22, 'P/B Ratio': 5}
    mock_get_features = MagicMock(return_value=mock_feature_dict)
    monkeypatch.setattr('src.scrapers.scrape_financial_ratios.get_financial_ratios_for_filing', mock_get_features)

    base_df = pd.DataFrame({
        'Ticker': ['NVDA', 'AMD'],
        'Filing Date': pd.to_datetime(['2023-01-05', '2023-02-10']).date,
        'Trade Date': pd.to_datetime(['2023-01-02', '2023-02-08']).date,
    })
    output_path = tmp_path / "financial_ratios.xlsx"

    # 2. Act: Run the main orchestrator
    scrape_all_financial_ratios(base_df, output_path, Path('/fake/edgar'))

    # 3. Assert: Check the results
    assert mock_get_features.call_count == 2
    assert output_path.exists()
    result_df = pd.read_excel(output_path)
    assert len(result_df) == 2
    assert 'P/E Ratio' in result_df.columns
