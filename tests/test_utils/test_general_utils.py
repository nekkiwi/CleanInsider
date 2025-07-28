import pandas as pd

from src.scrapers.scrape_feature_utils.general_utils import merge_and_save_features


def test_merge_and_save_features(test_data_dir):
    """
    Tests the feature merging logic by creating dummy stage files
    and verifying the merged output.
    """
    # 1. Arrange: Create dummy data and save to temporary files
    path1 = test_data_dir / "stage1.xlsx"
    path2 = test_data_dir / "stage2.xlsx"
    path3 = test_data_dir / "stage3.xlsx"
    output_path = test_data_dir / "final.xlsx"

    df1 = pd.DataFrame({'Ticker': ['AAPL'], 'Filing Date': ['2023-01-05'], 'Trade Date': ['2023-01-02'], 'Value': [1000]})
    df2 = pd.DataFrame({'Ticker': ['AAPL'], 'Filing Date': ['2023-01-05'], 'Trade Date': ['2023-01-02'], 'RSI': [50]})
    df3 = pd.DataFrame({'Ticker': ['AAPL'], 'Filing Date': ['2023-01-05'], 'Trade Date': ['2023-01-02'], 'P/E': [25]})

    df1.to_excel(path1, index=False)
    df2.to_excel(path2, index=False)
    df3.to_excel(path3, index=False)

    # 2. Act: Run the function to be tested
    merged_df = merge_and_save_features(path1, path2, path3, output_path)

    # 3. Assert: Check if the output is correct
    assert not merged_df.empty
    assert 'Value' in merged_df.columns
    assert 'RSI' in merged_df.columns
    assert 'P/E' in merged_df.columns
    assert len(merged_df) == 1
    assert output_path.exists()
