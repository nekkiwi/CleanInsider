import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("data")

@pytest.fixture
def sample_stock_df():
    """Creates a sample DataFrame of stock data for testing indicators."""
    data = {
        'Open': [100, 102, 101, 103, 105],
        'High': [103, 104, 103, 106, 106],
        'Low': [99, 101, 100, 102, 104],
        'Close': [102, 103, 102, 105, 105],
        'Volume': [1000, 1200, 1100, 1500, 1300]
    }
    index = pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'])
    return pd.DataFrame(data, index=index)
