# file: src/preprocess/utils.py

from pathlib import Path
import pandas as pd

def save_columns_list(df: pd.DataFrame, output_path: Path):
    """Saves the list of columns from a DataFrame to a text file."""
    with open(output_path, "w") as f:
        for col in df.columns:
            f.write(f"{col}\n")
    print(f"   -> Saved column list to {output_path}")
