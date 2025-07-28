import os
import pandas as pd
from pathlib import Path

def merge_and_save_features(config) -> pd.DataFrame:
    """Merges data from all stages using the new Ticker/Filing Date key."""
    try:
        df1 = pd.read_excel(config.STAGE_1_PATH)
        df2 = pd.read_excel(config.STAGE_2_PATH)
        df3 = pd.read_excel(config.STAGE_3_PATH)
    except FileNotFoundError as e:
        print(f"Error: Could not find a stage file for merging. {e}")
        return pd.DataFrame()

    # The new merge key, dropping Trade Date
    merge_keys = ['Ticker', 'Filing Date']
    for df in [df1, df2, df3]:
        df['Filing Date'] = pd.to_datetime(df['Filing Date'])

    merged_df = pd.merge(df1, df2, on=merge_keys, how='left')
    merged_df = pd.merge(merged_df, df3, on=merge_keys, how='left')
    
    # Do not save here, as we have a composite step next. Return the DataFrame.
    return merged_df

def create_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features from the fully merged dataset."""
    print("Creating final composite features...")
    
    # Value to MarketCap
    if 'Value' in df.columns and 'Market_Cap' in df.columns:
        df['Value_to_MarketCap'] = df['Value'] / df['Market_Cap']
        
    # CFO and President Buy Values (example)
    df['CFO_Buy_Value'] = df.apply(lambda row: row['Value'] if row['CFO'] == 1 else 0, axis=1)
    df['Pres_Buy_Value'] = df.apply(lambda row: row['Value'] if row['Pres'] == 1 else 0, axis=1)

    # Insider Importance Score (example scoring)
    df['Insider_Importance_Score'] = (df['CEO'] * 5 + df['CFO'] * 4 + df['Pres'] * 3 + df['VP'] * 2 + df['Dir'])
    
    # Final column ordering to match user request as closely as possible
    # (This is a best-effort reordering, some columns might be missing if scraping failed)
    all_requested_cols = [
        'Ticker', 'Filing Date', 'Number_of_Purchases', 'Price', 'Qty', 'Owned', 'dOwn', 'Value', 
        'CEO', 'CFO', 'Dir', 'Pres', 'VP', 'TenPercent', 'Days_Since_Trade', 'RSI_14', 
        'MACD_Signal', 'MACD_Hist', 'ADX_14', 'CCI_14', 'ROC', 'MFI_14', 'STOCH_D', 
        'Bollinger_Lower', 'OBV', 'Beta_x', 'Jensen_Alpha', 'Tracking_Error', 'Information_Ratio', 
        'Net_Profit_Margin', 'Investing_Cash_Flow', 'Financing_Cash_Flow', 'Free_Cash_Flow', 
        'Market_Cap', 'Price_to_Earnings_Ratio', 'Price_to_Book_Ratio', 'Price_to_Sales_Ratio', 
        'Operating_Cash_Flow_to_Market_Cap', 'Net_Income_to_Market_Cap', 'EPS', 'Beta_y', 
        '52_Week_Low_Normalized', 'Days_Since_IPO', 'VIX_Close', 'VIX_SMA50', 'SP500_Above_SMA50', 
        'SP500_Above_SMA200', 'Sector_Consumer Cyclical', 'Sector_Financial Services', 
        'Sector_Healthcare', 'Sector_Industrials', 'Sector_Technology', 'CFO_Buy_Value', 
        'Pres_Buy_Value', 'Insider_Importance_Score', 'Value_to_MarketCap', 'Distance_from_52W_High', 
        'Day_Of_Year', 'Day_Of_Quarter'
    ]
    
    # Use only the columns that actually exist in the final dataframe
    final_cols = [col for col in all_requested_cols if col in df.columns]
    return df[final_cols]
def create_output_directories(paths: list[Path]):
    """
    Creates directories if they do not already exist.

    Args:
        paths (list[Path]): A list of Path objects for the directories to create.
    """
    for path in paths:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {path}: {e}")
            raise

def report_missing_data(df: pd.DataFrame):
    """
    Calculates and prints the percentage of missing values for each column.

    Args:
        df (pd.DataFrame): The dataframe to analyze.
    """
    if df.empty:
        print("Cannot report on an empty dataframe.")
        return
        
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_report = missing_percentage[missing_percentage > 0].sort_values(ascending=False)

    if missing_report.empty:
        print("No missing data found in any columns. Excellent!")
    else:
        print("Percentage of empty rows per feature column:")
        # Use to_string() to ensure the full report is printed without truncation
        print(missing_report.to_string())