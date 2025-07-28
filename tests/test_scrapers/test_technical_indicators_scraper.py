from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from src.scrapers.scrape_technical_indicators import (
    get_technical_indicators_for_filing,
    scrape_all_technical_indicators,
)

# This test file focuses on the logic within scrape_technical_indicators.py
# It mocks the lower-level functions from the utils file.

def test_get_technical_indicators_for_filing_local_success(monkeypatch):
    """
    Tests the success path where the primary method (local Stooq data) returns valid features.
    It should NOT call the yfinance fallback.
    """
    # 1. Arrange: Mock the utility functions
    mock_local_data = pd.DataFrame([{'RSI': 55, 'MACD': 0.8}])
    mock_get_local = MagicMock(return_value=mock_local_data)
    mock_get_yf = MagicMock(side_effect=AssertionError("Fallback yfinance function should not be called"))

    monkeypatch.setattr('src.scrapers.scrape_feature_utils.technical_indicator_utils.get_techn_ind_local', mock_get_local)
    monkeypatch.setattr('src.scrapers.scrape_feature_utils.technical_indicator_utils.get_tech_ind_yf', mock_get_yf)

    # 2. Act: Run the function
    features = get_technical_indicators_for_filing('AAPL', pd.to_datetime('2023-01-10'), Path('/fake/stooq'))

    # 3. Assert: Check the results and that the correct functions were called
    mock_get_local.assert_called_once()
    assert features['RSI'] == 55
    assert 'MACD' in features

def test_get_technical_indicators_for_filing_fallback_to_yf(monkeypatch):
    """
    Tests the fallback path where the local method fails (returns empty)
    and the secondary method (yfinance) succeeds.
    """
    # 1. Arrange: Mock the utility functions
    mock_get_local = MagicMock(return_value=pd.DataFrame()) # Fails by returning empty DF
    mock_yf_data = pd.DataFrame([{'RSI': 65, 'MACD': -0.2}])
    mock_get_yf = MagicMock(return_value=mock_yf_data)

    monkeypatch.setattr('src.scrapers.scrape_feature_utils.technical_indicator_utils.get_techn_ind_local', mock_get_local)
    monkeypatch.setattr('src.scrapers.scrape_feature_utils.technical_indicator_utils.get_tech_ind_yf', mock_get_yf)

    # 2. Act: Run the function
    features = get_technical_indicators_for_filing('MSFT', pd.to_datetime('2023-03-15'), Path('/fake/stooq'))

    # 3. Assert: Check the results and function calls
    mock_get_local.assert_called_once()
    mock_get_yf.assert_called_once()
    assert features['RSI'] == 65

def test_get_technical_indicators_for_filing_all_fail(monkeypatch):
    """
    Tests the failure path where both local and yfinance methods fail.
    It should return an empty dictionary.
    """
    # 1. Arrange: Mock both utility functions to fail
    mock_get_local = MagicMock(return_value=pd.DataFrame())
    mock_get_yf = MagicMock(return_value=pd.DataFrame())

    monkeypatch.setattr('src.scrapers.scrape_feature_utils.technical_indicator_utils.get_techn_ind_local', mock_get_local)
    monkeypatch.setattr('src.scrapers.scrape_feature_utils.technical_indicator_utils.get_tech_ind_yf', mock_get_yf)

    # 2. Act: Run the function
    features = get_technical_indicators_for_filing('TSLA', pd.to_datetime('2023-05-20'), Path('/fake/stooq'))

    # 3. Assert: The result should be an empty dictionary
    assert features == {}

def test_scrape_all_technical_indicators_integration(monkeypatch, tmp_path):
    """
    Tests the main orchestrator function to ensure it processes a DataFrame,
    calls the underlying logic correctly, and saves the output.
    """
    # 1. Arrange: Mock the high-level logic function and prepare data
    mock_feature_dict = {'RSI': 70, 'MACD': 1.2}
    mock_get_features = MagicMock(return_value=mock_feature_dict)
    monkeypatch.setattr('src.scrapers.scrape_technical_indicators.get_technical_indicators_for_filing', mock_get_features)

    base_df = pd.DataFrame({
        'Ticker': ['NVDA', 'AMD', 'NVDA'],
        'Filing Date': pd.to_datetime(['2023-01-05', '2023-02-10', '2023-01-05']).date,
        'Trade Date': pd.to_datetime(['2023-01-02', '2023-02-08', '2023-01-02']).date,
    })
    output_path = tmp_path / "tech_indicators.xlsx"

    # 2. Act: Run the main orchestrator
    scrape_all_technical_indicators(base_df, output_path, Path('/fake/stooq'))

    # 3. Assert: Check the results
    # It should only be called for unique Ticker/Filing Date pairs
    assert mock_get_features.call_count == 2
    assert output_path.exists()

    # Verify the contents of the saved file
    result_df = pd.read_excel(output_path)
    assert len(result_df) == 2
    assert 'RSI' in result_df.columns
    assert 'Ticker' in result_df.columns
