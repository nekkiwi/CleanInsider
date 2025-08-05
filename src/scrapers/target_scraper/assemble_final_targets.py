# file: src/scrapers/target_scraper/assemble_final_targets.py

import pandas as pd
from pathlib import Path

def assemble_final_targets(config, n_splits: int):
    """
    Merges pre-calculated master targets and saves them to a clean, parallel
    directory structure under data/scrapers/targets/.
    """
    print("\n--- STEP 3: Assembling Final Targets into Label Files ---")
    
    # Input directories
    features_dir = Path(config.FEATURES_OUTPUT_PATH)
    targets_dir = Path(config.TARGETS_OUTPUT_PATH)
    
    # Path to the single source of truth for all calculated targets
    master_targets_path = targets_dir / "master_targets.parquet"

    if not master_targets_path.exists():
        raise FileNotFoundError(f"Master targets file not found at {master_targets_path}. Please run step 2 first.")
    
    master_targets_df = pd.read_parquet(master_targets_path)
    master_targets_df['Filing Date'] = pd.to_datetime(master_targets_df['Filing Date'])

    # Helper function to perform the merge and save to the specified label path
    def create_label_file(feature_path: Path, label_path: Path):
        """Reads feature identifiers, merges targets, and saves to the label path."""
        if not feature_path.exists():
            print(f"  - Skipping: Feature file not found at {feature_path}")
            return
            
        df_features = pd.read_parquet(feature_path)
        df_features['Filing Date'] = pd.to_datetime(df_features['Filing Date'])
        
        # Merge the identifiers from the feature set with the master target list
        # This ensures the labels perfectly align with the features.
        df_labels = pd.merge(
            df_features[['Ticker', 'Filing Date']], 
            master_targets_df, 
            on=['Ticker', 'Filing Date'], 
            how='inner'
        )
        
        # Ensure the output directory exists before saving
        label_path.parent.mkdir(parents=True, exist_ok=True)
        df_labels.to_parquet(label_path, index=False)
        print(f"✅ Saved final labels to {label_path}")

    # --- Assemble for each training and validation fold ---
    for i in range(1, n_splits + 1):
        # Define paths for training data
        train_features_path = features_dir / f"fold_{i}" / "training_data.parquet"
        train_labels_path = targets_dir / f"fold_{i}" / "training_labels.parquet"
        create_label_file(train_features_path, train_labels_path)
        
        # Define paths for validation data
        val_features_path = features_dir / f"fold_{i}" / "validation_data.parquet"
        val_labels_path = targets_dir / f"fold_{i}" / "validation_labels.parquet"
        create_label_file(val_features_path, val_labels_path)

    # --- Assemble for the final test set ---
    test_features_path = features_dir / "test_set" / "test_data.parquet"
    # Save the test labels in the `targets` directory, parallel to the fold folders
    test_labels_path = targets_dir / "test_set" / "test_labels.parquet"
    create_label_file(test_features_path, test_labels_path)

