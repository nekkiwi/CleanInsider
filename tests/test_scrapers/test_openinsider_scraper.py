from unittest.mock import MagicMock
from bs4 import BeautifulSoup
from src.scrapers.scrape_openinsider import scrape_openinsider

def test_scrape_openinsider_parsing(mocker, test_data_dir):
    """
    Tests if the OpenInsider scraper correctly parses a mocked HTML table.
    It does not make a real network request.
    """
    # 1. Arrange: Create mock HTML content and mock the requests.get call
    mock_html = """
    <html><body>
    <table class="tinytable">
      <tr><th>Filing Date</th><th>Trade Date</th><th>Ticker</th><th>Value</th></tr>
      <tr><td>2023-01-05 18:00:00</td><td>2023-01-03</td><td><a href="">TEST</a></td><td>$1,000</td></tr>
    </table>
    </body></html>
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = mock_html
    mocker.patch("requests.get", return_value=mock_response)

    output_path = test_data_dir / "openinsider.xlsx"

    # 2. Act: Run the scraper
    df, _ = scrape_openinsider(num_weeks=1, output_path=output_path)

    # 3. Assert: Check if the dataframe was parsed correctly
    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]['Ticker'] == 'TEST'
    assert df.iloc[0]['Value'] == 1000  # Check for correct numeric conversion
