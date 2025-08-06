# file: src/preprocess/utils.py

import joblib
import json
from pathlib import Path
import pandas as pd

def save_columns_list(df: pd.DataFrame, output_path: Path):
    """Saves the list of columns from a DataFrame to a text file."""
    with open(output_path, "w") as f:
        for col in df.columns:
            f.write(f"{col}\n")
    print(f"   -> Saved column list to {output_path}")

def save_preprocessing_artifacts(processor, fold_dir: Path):
    """Save all preprocessing artifacts for a specific fold."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Save fitted StandardScaler
    scaler_path = fold_dir / "scaler.pkl"
    joblib.dump(processor.scaler, scaler_path)
    
    # Save outlier bounds from apply_additional_preprocessing
    outlier_path = fold_dir / "outlier_bounds.json"
    with outlier_path.open("w") as f:
        json.dump(processor.outlier_bounds, f)
    
    # Save imputation values for unscaled columns
    imputation_path = fold_dir / "imputation_values.json"
    if hasattr(processor, 'imputation_values_for_unscaled') and processor.imputation_values_for_unscaled is not None:
        imputation_dict = processor.imputation_values_for_unscaled.dropna().to_dict()
        with imputation_path.open("w") as f:
            json.dump(imputation_dict, f)
    
    # Save columns information
    columns_info = {
        'columns_to_scale': list(processor.columns_to_scale) if processor.columns_to_scale else [],
        'final_columns': list(processor.final_columns) if processor.final_columns else [],
        'columns_to_drop_missing': list(processor.columns_to_drop_missing) if hasattr(processor, 'columns_to_drop_missing') else [],
        'columns_to_drop_variance': list(processor.columns_to_drop_variance) if processor.columns_to_drop_variance else [],
        'columns_to_drop_corr': list(processor.columns_to_drop_corr) if processor.columns_to_drop_corr else []
    }
    
    columns_path = fold_dir / "preprocessing_columns.json"
    with columns_path.open("w") as f:
        json.dump(columns_info, f, indent=2)
    
    print(f"[SAVE-ARTIFACTS] Saved preprocessing artifacts to {fold_dir}")

def load_preprocessing_artifacts(fold_dir: Path):
    """Load all preprocessing artifacts for a specific fold."""
    if not fold_dir.exists():
        raise FileNotFoundError(f"Preprocessing artifacts directory not found: {fold_dir}")
    
    # Load fitted StandardScaler
    scaler_path = fold_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    
    # Load outlier bounds
    outlier_path = fold_dir / "outlier_bounds.json"
    outlier_bounds = {}
    if outlier_path.exists():
        with outlier_path.open("r") as f:
            outlier_bounds = json.load(f)
    
    # Load imputation values
    imputation_path = fold_dir / "imputation_values.json"
    imputation_values = {}
    if imputation_path.exists():
        with imputation_path.open("r") as f:
            imputation_values = json.load(f)
    
    # Load columns information
    columns_path = fold_dir / "preprocessing_columns.json"
    columns_info = {}
    if columns_path.exists():
        with columns_path.open("r") as f:
            columns_info = json.load(f)
    
    return scaler, outlier_bounds, imputation_values, columns_info
