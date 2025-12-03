#!/usr/bin/env python
"""
Upload trained models to Google Drive.

Supports uploading multiple model strategies for parallel inference.

Usage:
    python scripts/upload_models_to_drive.py --strategies 1w_tp0p05_sl-0p05 2w_tp0p05_sl-0p05 1m_tp0p05_sl-0p05
    python scripts/upload_models_to_drive.py --dry-run
    python scripts/upload_models_to_drive.py --list  # List available strategies
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import config


def list_available_strategies(models_path: Path) -> list:
    """List all available model strategies."""
    strategies = []
    for d in sorted(models_path.iterdir()):
        if d.is_dir():
            # Count files
            pkl_files = list(d.rglob("*.pkl"))
            if pkl_files:
                strategies.append({
                    "name": d.name,
                    "files": len(pkl_files),
                    "size_mb": sum(f.stat().st_size for f in pkl_files) / (1024 * 1024)
                })
    return strategies


def get_strategy_files(strategy_path: Path) -> dict:
    """Get all model files for a strategy in a flat structure."""
    files = {
        "classifiers": [],
        "regressors": [],
        "metadata": []
    }
    
    for fold_dir in sorted(strategy_path.iterdir()):
        if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
            fold_num = fold_dir.name.split("_")[1]
            
            for seed_dir in sorted(fold_dir.iterdir()):
                if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                    seed_num = seed_dir.name.split("_")[1]
                    
                    # Collect files with flattened naming
                    for f in seed_dir.iterdir():
                        if f.suffix == ".pkl":
                            new_name = f"fold{fold_num}_seed{seed_num}_{f.name}"
                            if "classifier" in f.name:
                                files["classifiers"].append((f, new_name))
                            elif "regressor" in f.name:
                                files["regressors"].append((f, new_name))
                            elif "metadata" in f.name:
                                files["metadata"].append((f, new_name))
    
    return files


def get_preprocessing_files(preprocessing_path: Path) -> list:
    """Get preprocessing artifact files."""
    files = []
    
    # Common features (root level)
    common_features = preprocessing_path / "common_features.json"
    if common_features.exists():
        files.append((common_features, "common_features.json"))
    
    # Fold 5 artifacts (or any available fold)
    for fold_num in [5, 4, 3, 2, 1]:
        fold_dir = preprocessing_path / f"fold_{fold_num}"
        if fold_dir.exists():
            for f in fold_dir.iterdir():
                if f.is_file():
                    files.append((f, f.name))
            break
    
    return files


def upload_to_drive(gdrive_client, files: list, folder_id: str, subfolder: str = None, dry_run: bool = False) -> int:
    """Upload files to Google Drive."""
    uploaded = 0
    
    for src_path, dest_name in files:
        dest_path = f"{subfolder}/{dest_name}" if subfolder else dest_name
        
        if dry_run:
            print(f"  [DRY RUN] Would upload: {dest_path}")
            uploaded += 1
        else:
            result = gdrive_client.upload_file(src_path, folder_id, dest_path)
            if result:
                uploaded += 1
                print(f"  Uploaded: {dest_path}")
            else:
                print(f"  [FAIL] Failed to upload: {dest_path}")
    
    return uploaded


def main():
    parser = argparse.ArgumentParser(description="Upload models to Google Drive")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["1w_tp0p05_sl-0p05"],
        help="Strategy folders to upload (space-separated)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available strategies and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading"
    )
    
    args = parser.parse_args()
    
    models_path = config.MODELS_PATH
    preprocessing_path = config.PREPROCESSING_ARTIFACTS_PATH
    
    if not models_path.exists():
        print(f"[ERROR] Models directory not found: {models_path}")
        sys.exit(1)
    
    # List mode
    if args.list:
        print("=" * 60)
        print("AVAILABLE MODEL STRATEGIES")
        print("=" * 60)
        
        strategies = list_available_strategies(models_path)
        
        for s in strategies:
            print(f"  {s['name']:25s}  {s['files']:3d} files  {s['size_mb']:6.2f} MB")
        
        print(f"\nTotal: {len(strategies)} strategies")
        print("\nRecommended for multi-model:")
        print("  --strategies 1w_tp0p05_sl-0p05 2w_tp0p05_sl-0p05 1m_tp0p05_sl-0p05")
        return
    
    print("=" * 60)
    print("UPLOAD MODELS TO GOOGLE DRIVE")
    print("=" * 60)
    print(f"\nStrategies to upload: {', '.join(args.strategies)}")
    
    # Validate strategies exist
    for strategy in args.strategies:
        strategy_path = models_path / strategy
        if not strategy_path.exists():
            print(f"[ERROR] Strategy not found: {strategy}")
            print(f"  Available: {[s['name'] for s in list_available_strategies(models_path)]}")
            sys.exit(1)
    
    # Summary
    print("\n--- Files to Upload ---")
    total_files = 0
    total_size = 0
    
    for strategy in args.strategies:
        strategy_path = models_path / strategy
        files = get_strategy_files(strategy_path)
        num_files = sum(len(v) for v in files.values())
        size_mb = sum(f[0].stat().st_size for v in files.values() for f in v) / (1024 * 1024)
        total_files += num_files
        total_size += size_mb
        print(f"  {strategy}: {num_files} files ({size_mb:.2f} MB)")
    
    # Preprocessing
    preproc_files = get_preprocessing_files(preprocessing_path)
    preproc_size = sum(f[0].stat().st_size for f in preproc_files) / (1024 * 1024)
    total_files += len(preproc_files)
    total_size += preproc_size
    print(f"  preprocessing: {len(preproc_files)} files ({preproc_size:.2f} MB)")
    
    print(f"\nTotal: {total_files} files, {total_size:.2f} MB")
    
    if args.dry_run:
        print("\n[DRY RUN MODE]")
    
    # Initialize Google Drive client
    from src.alpaca.google_drive import GoogleDriveClient
    gdrive_client = GoogleDriveClient()
    
    if not gdrive_client.is_connected():
        print("\n[ERROR] Google Drive not connected")
        sys.exit(1)
    
    if not gdrive_client.models_folder_id:
        print("\n[ERROR] GDRIVE_MODELS_FOLDER_ID not set")
        sys.exit(1)
    
    # Upload each strategy
    print("\n--- Uploading ---")
    
    for strategy in args.strategies:
        print(f"\n[{strategy}]")
        strategy_path = models_path / strategy
        files = get_strategy_files(strategy_path)
        
        # Flatten all files for this strategy
        all_files = files["classifiers"] + files["regressors"] + files["metadata"]
        
        uploaded = upload_to_drive(
            gdrive_client, 
            all_files, 
            gdrive_client.models_folder_id,
            subfolder=strategy,
            dry_run=args.dry_run
        )
        print(f"  -> {uploaded}/{len(all_files)} files")
    
    # Upload preprocessing
    print("\n[preprocessing]")
    uploaded = upload_to_drive(
        gdrive_client,
        preproc_files,
        gdrive_client.models_folder_id,
        subfolder="preprocessing",
        dry_run=args.dry_run
    )
    print(f"  -> {uploaded}/{len(preproc_files)} files")
    
    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE" if not args.dry_run else "DRY RUN COMPLETE")
    print("=" * 60)
    
    # Print folder structure for reference
    print("\nExpected Google Drive structure:")
    print(f"  {gdrive_client.models_folder_id}/")
    for strategy in args.strategies:
        print(f"    {strategy}/")
        print(f"      fold1_seed42_classifier.pkl")
        print(f"      fold1_seed42_regressor.pkl")
        print(f"      fold1_seed42_metadata.pkl")
        print(f"      ... (75 files total)")
    print(f"    preprocessing/")
    print(f"      common_features.json")
    print(f"      scaler.pkl")
    print(f"      imputation_values.json")
    print(f"      outlier_bounds.json")
    print(f"      preprocessing_columns.json")


if __name__ == "__main__":
    main()
