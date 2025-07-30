# file: load_macro_features.py (Corrected and Robust)

from functools import lru_cache
from pathlib import Path

import pandas as pd

# --- COMPLETE Macro Code Map (with duplicates resolved) ---
MACRO_CODE_MAP = {
    "adpeus": "ADP_Employment_Change_K",
    "avheus": "Avg_Hourly_Earnings_YoY_Pct",
    "avwhus": "Avg_Weekly_Hours",
    "bsimus": "Business_Inventories_MoM_Pct",
    "bsiyus": "Business_Inventories_YoY_Pct",
    "clinus": "Composite_Leading_Indicators",
    "cncius": "Consumer_Confidence_Index",
    "cncrus": "Consumer_Credit",
    "cpcmus": "Core_CPI_MoM_Change_Pct",
    "cpcyus": "Core_CPI_YoY_Change_Pct",
    "cpimus": "CPI_MoM_Change_Pct",
    "cpiyus": "CPI_YoY_Pct",
    "cpumus": "Core_CPI_Index_Value",
    "crabus": "Car_Registrations",
    "ctclus": "Continuing_Jobless_Claims",
    "ctspus": "Corporate_Profits_QoQ_Pct",
    "dglmus": "Durable_Goods_Orders_MoM_Pct",
    "dgomus": "Factory_Orders_MoM_Pct",
    "emcius": "Employment_Cost_Index",
    "esmius": "Economic_Sentiment",
    "expmus": "Exports_MoM_Pct",
    "exprus": "Exports_Real",
    "expyus": "Exports_YoY_Pct",
    "fdphus": "Federal_Debt_Held_by_Public",
    "fdrhus": "Federal_Debt_to_GDP_Ratio",
    "fgomus": "Federal_Government_Orders",
    "gddqus": "GDP_Deflator_QoQ_Pct",
    "gdpqus": "GDP_QoQ_Pct",
    "gdpyus": "GDP_YoY_Pct",
    "gvbgus": "Government_Budget",
    "hbpmus": "Building_Permits_MoM_Pct",
    "hoesus": "Existing_Home_Sales",
    "honsus": "New_Home_Sales_K",
    "hopmus": "House_Price_Index_MoM_Pct",
    "hopqus": "House_Price_Index_QoQ_Pct",
    "hosmus": "Housing_Starts_K",
    "imprus": "Imports_MoM_Pct",
    "injcus": "Initial_Jobless_Claims",
    "inpmus": "Industrial_Production_MoM_Pct",
    "inrtus": "Federal_Funds_Rate",
    "ipimus": "Industrial_Production_Index",
    "ipiyus": "Industrial_Production_YoY_Pct",  # This is the primary one we will keep
    "ismnus": "ISM_Manufacturing_PMI",
    "isnfus": "ISM_Services_PMI",
    "ldiius": "Leading_Economic_Index",
    "mfpmus": "Manufacturing_Payrolls",
    "nahbus": "NAHB_Housing_Market_Index",
    "nginus": "Natural_Gas_Inventories",
    "nfpmus": "NonFarm_Payrolls_Change_K",
    "pcdmus": "Personal_Consumption_Expenditures_MoM_Pct",
    "pcdyus": "Personal_Consumption_Expenditures_YoY_Pct",
    "pcequs": "PCE_Price_Index_QoQ_Pct",
    "pceyus": "PCE_Price_Index_YoY_Pct",
    "pecyus": "Core_PCE_Price_Index_YoY_Pct",
    "pmchus": "Chicago_PMI",
    "pmcpus": "Philadelphia_Fed_Manufacturing_Index",
    "pmmnus": "NY_Empire_State_Manufacturing_Index",
    "pmsrus": "Personal_Saving_Rate",
    "ppimus": "PPI_MoM_Pct",
    "ppiyus": "PPI_Final_Demand_YoY_Pct",
    "psimus": "Personal_Income_MoM_Pct",
    "pssmus": "Personal_Spending_MoM_Pct",
    "rsamus": "Retail_Sales_MoM_Pct",
    "rsayus": "Retail_Sales_YoY_Pct",
    "rslmus": "Retail_Sales_Less_Autos_MoM_Pct",
    "rslyus": "Retail_Sales_Less_Autos_YoY_Pct",
    "s20yus": "SP_CaseShiller_Home_Price_20City_YoY_Pct",
    "tnlfus": "Total_Nonfarm_Labor_Force_K",
    "tntfus": "Total_Trade_Balance",
    "trbnus": "Trade_Balance",
    "ulcqus": "Unit_Labor_Costs_QoQ_Pct",
    "ulcyus": "Unit_Labor_Costs_YoY_Pct",
    "umccus": "Consumer_Sentiment_UMich",
    "unrtus": "Unemployment_Rate",
    "whimus": "Wholesale_Inventories_MoM_Pct",
    "whiyus": "Wholesale_Inventories_YoY_Pct",
    "whsmus": "Wholesale_Sales_MoM_Pct",
    "whsyus": "Wholesale_Sales_YoY_Pct",
}


@lru_cache(maxsize=1)
def load_all_macro_data(macro_us_path: Path) -> pd.DataFrame:
    if not macro_us_path.exists():
        raise FileNotFoundError(f"Macro data directory not found at '{macro_us_path}'")
    print(f"🔄 Loading all macro data from '{macro_us_path}'...")

    # --- THIS IS THE FIX ---
    # Use a dictionary to store Series, which automatically handles duplicate feature names.
    macro_series_dict = {}

    for filepath in macro_us_path.glob("*.txt"):
        try:
            df = pd.read_csv(filepath, header=0)
            date_col_name = next(
                (col for col in df.columns if "DATE" in col.upper()), None
            )
            value_col_name = next(
                (col for col in df.columns if "OPEN" in col.upper()), None
            )
            if not date_col_name or not value_col_name:
                continue

            df["parsed_date"] = pd.to_datetime(df[date_col_name], errors="coerce")
            df["parsed_value"] = pd.to_numeric(df[value_col_name], errors="coerce")
            df.dropna(subset=["parsed_date", "parsed_value"], inplace=True)
            df.set_index("parsed_date", inplace=True)
            if df.empty:
                continue

            file_code = filepath.stem.split(".")[0]
            feature_name = MACRO_CODE_MAP.get(file_code, f"MACRO_{file_code.upper()}")

            # If this feature name is already in our dictionary, skip it.
            if feature_name in macro_series_dict:
                continue

            macro_series_dict[feature_name] = df["parsed_value"]

        except Exception:
            continue

    if not macro_series_dict:
        print("   ❌ No macro data could be loaded. Check file formats.")
        return pd.DataFrame()

    # Convert the dictionary of Series to a list of Series to be concatenated
    all_macro_series = list(macro_series_dict.values())
    combined_df = pd.concat(all_macro_series, axis=1, keys=macro_series_dict.keys())
    combined_df.ffill(inplace=True)

    print(
        f"   ✅ Successfully loaded and combined {len(combined_df.columns)} unique macro indicators."
    )
    return combined_df.sort_index()


def load_macro_feature_df(dates_list: list, stooq_db_dir: str) -> pd.DataFrame:
    stooq_macro_path = Path(stooq_db_dir) / "macro" / "us"
    master_macro_df = load_all_macro_data(stooq_macro_path)
    if master_macro_df.empty:
        print("❌ CRITICAL: master_macro_df is empty after loading. Stopping.")
        return pd.DataFrame()

    print("\n--- Starting Optimized Lookup Phase ---")

    query_dates_series = pd.to_datetime(pd.Series(dates_list, name="Query_Date"))
    target_dates_series = (query_dates_series.dt.to_period("M") - 1).dt.to_timestamp(
        "M"
    )
    unique_target_dates = target_dates_series.unique()

    print(
        f"   Optimized: performing {len(unique_target_dates)} lookups instead of {len(dates_list)}."
    )

    results_cache = {
        target_date: master_macro_df.asof(target_date)
        for target_date in unique_target_dates
    }

    print("   Mapping cached results back to original dates...")

    macro_data_rows = [results_cache[d] for d in target_dates_series]
    final_df = pd.DataFrame(macro_data_rows)
    final_df["Query_Date"] = query_dates_series.dt.strftime("%Y-%m-%d").values

    cols = ["Query_Date"] + [col for col in final_df.columns if col != "Query_Date"]
    print("--- ✅ Optimized Lookup Phase Complete ---")
    return final_df[cols]


if __name__ == "__main__":
    STOOQ_DB_DIR = "../../data/stooq_database"
    OUTPUT_CSV = "macro_features_summary_fast.csv"

    dates_to_process = [
        "2024-01-15",
        "2024-01-20",
        "2024-02-10",
        "2024-03-25",
        "2024-03-30",
        "2024-04-05",
    ]
    print(f"Input Dates: {dates_to_process}")

    macro_feature_df = load_macro_feature_df(dates_to_process, STOOQ_DB_DIR)

    if not macro_feature_df.empty:
        macro_feature_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 Macro features summary saved to '{OUTPUT_CSV}'")
        print("\nFinal DataFrame sample:")
        print(macro_feature_df)
    else:
        print("\nNo data was generated, so no file was saved.")
