import os
from pathlib import Path

import pandas as pd


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


def report_missing_data(df: pd.DataFrame, output_dir: Path = None):
    """
    Calculates and prints the percentage of missing values for each column.
    If an output_dir is provided, it saves the report to a text file.

    Args:
        df (pd.DataFrame): The dataframe to analyze.
        output_dir (Path, optional): The directory to save the report in.
                                     Defaults to None.
    """
    if df.empty:
        print("Cannot report on an empty dataframe.")
        return

    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_report = missing_percentage[missing_percentage >= 0].sort_values(
        ascending=False
    )

    report_string = ""
    if missing_report.empty:
        report_string = "No missing data found in any columns. Excellent!"
    else:
        # Use to_string() to ensure the full report is captured without truncation
        header = "Percentage of empty rows per feature column:\n"
        report_string = header + missing_report.to_string()

    # Always print the report to the console for immediate feedback
    # print(report_string)

    # Save the report to a file if an output directory was provided
    if output_dir:
        # Ensure the directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "info" / "missing_data_report.txt"
        try:
            with open(report_path, "w") as f:
                f.write(report_string)
            print(f"\n💾 Missing data report saved to: {report_path}")
        except Exception as e:
            print(f"\n❌ Could not save missing data report: {e}")


def create_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features that depend on data from multiple sources after the merge.
    """
    df_comp = df.copy()

    # Insider Importance Score: Weighted sum of roles multiplied by the trade value
    df_comp["Insider_Importance_Score"] = (
        df_comp["CEO"] * 3
        + df_comp["CFO"] * 3
        + df_comp["Pres"] * 2
        + df_comp["Dir"] * 1
        + df_comp["VP"] * 1
        + df_comp["TenPercent"] * 0.5
    ) * df_comp["Value"]

    # Role-specific buy values
    df_comp["CFO_Buy_Value"] = df_comp["Value"] * df_comp["CFO"]
    df_comp["Pres_Buy_Value"] = df_comp["Value"] * df_comp["Pres"]

    # Value to Market Cap Ratio
    if "Market_Cap" in df_comp.columns and "Value" in df_comp.columns:
        # Use .loc to avoid SettingWithCopyWarning on a filtered view
        valid_rows = (df_comp["Market_Cap"].notna()) & (df_comp["Market_Cap"] > 0)
        df_comp.loc[valid_rows, "Value_to_MarketCap"] = (
            df_comp.loc[valid_rows, "Value"] / df_comp.loc[valid_rows, "Market_Cap"]
        )

    return df_comp


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates date-based features from the 'Filing Date' column
    and adds them to the DataFrame.

    This function is designed to be called from the scrapers.

    Args:
        df (pd.DataFrame): DataFrame with a 'Filing Date' column of type datetime.

    Returns:
        pd.DataFrame: The DataFrame with 'Day_Of_Year' and 'Day_Of_Quarter' added.
    """
    # Ensure 'Filing Date' is a datetime object before using the .dt accessor
    filing_date_series = pd.to_datetime(df["Filing Date"], errors="coerce")

    df["Day_Of_Year"] = filing_date_series.dt.dayofyear
    q_start_dates = filing_date_series.dt.to_period("Q").apply(lambda p: p.start_time)
    df["Day_Of_Quarter"] = (filing_date_series - q_start_dates).dt.days + 1

    return df
