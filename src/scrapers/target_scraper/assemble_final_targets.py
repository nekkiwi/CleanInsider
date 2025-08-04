# file: src/scrapers/target_scraper/3_assemble_final_targets.py
from pathlib import Path

import pandas as pd


def assemble_final_targets(config, n_splits: int):
    """
    Merges the pre-calculated master targets into each fold's final target file.
    """
    print("\n--- STEP 3: Assembling Final Targets for Each Fold ---")
    features_dir = Path(config.FEATURES_OUTPUT_PATH)
    targets_dir = Path(config.TARGETS_OUTPUT_PATH)
    master_targets_path = targets_dir / "master_targets.parquet"

    if not master_targets_path.exists():
        raise FileNotFoundError(
            f"Master targets file not found at {master_targets_path}. Please run step 2 first."
        )

    master_targets_df = pd.read_parquet(master_targets_path)
    master_targets_df["Filing Date"] = pd.to_datetime(master_targets_df["Filing Date"])

    # Assemble for each training fold
    for i in range(1, n_splits):
        fold_features_path = features_dir / f"fold_{i}" / "preprocessed_fold.parquet"
        df_fold_features = pd.read_parquet(fold_features_path)
        df_fold_features["Filing Date"] = pd.to_datetime(
            df_fold_features["Filing Date"]
        )

        # Merge the full set of calculated targets
        df_fold_targets = pd.merge(
            df_fold_features[["Ticker", "Filing Date"]],
            master_targets_df,
            on=["Ticker", "Filing Date"],
            how="inner",
        )

        fold_targets_dir = targets_dir / f"fold_{i}"
        fold_targets_dir.mkdir(parents=True, exist_ok=True)
        df_fold_targets.to_parquet(fold_targets_dir / "targets.parquet", index=False)
        print(f"✅ Saved final targets for Fold {i}.")

    # Assemble for the test set
    test_set_path = features_dir / "final_test_set_unprocessed.parquet"
    df_test = pd.read_parquet(test_set_path)
    df_test["Filing Date"] = pd.to_datetime(df_test["Filing Date"])
    df_test_targets = pd.merge(
        df_test[["Ticker", "Filing Date"]],
        master_targets_df,
        on=["Ticker", "Filing Date"],
        how="inner",
    )
    df_test_targets.to_parquet(targets_dir / "test_set_targets.parquet", index=False)
    print("✅ Saved final targets for the test set.")


if __name__ == "__main__":
    from src import config

    assemble_final_targets(config, n_splits=7)
