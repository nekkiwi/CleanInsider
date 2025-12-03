#!/usr/bin/env python
"""
Prepare models for manual upload to Google Drive.

Creates a clean directory structure in 'deploy/' that can be dragged to Drive.
"""

import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import config

# Configuration - edit these to choose your 3 models
STRATEGIES = [
    ("1w_tp0p05_sl-0p05", "model_1w_tp5_sl5"),   # 1 week, 5% TP, -5% SL
    ("2w_tp0p05_sl-0p05", "model_2w_tp5_sl5"),   # 2 weeks, 5% TP, -5% SL  
    ("1m_tp0p05_sl-0p05", "model_1m_tp5_sl5"),   # 1 month, 5% TP, -5% SL
]

def main():
    models_path = config.MODELS_PATH
    preprocessing_path = config.PREPROCESSING_ARTIFACTS_PATH
    deploy_path = project_root / "deploy"
    
    # Clean and create deploy directory
    if deploy_path.exists():
        print(f"Removing existing deploy directory...")
        shutil.rmtree(deploy_path)
    
    deploy_path.mkdir(parents=True)
    print(f"Created: {deploy_path}")
    
    # Process each strategy
    for src_name, dest_name in STRATEGIES:
        print(f"\n=== {dest_name} ({src_name}) ===")
        
        src_strategy = models_path / src_name
        if not src_strategy.exists():
            print(f"  [ERROR] Strategy not found: {src_strategy}")
            continue
        
        # Create model directory structure
        model_dir = deploy_path / dest_name
        weights_dir = model_dir / "weights"
        preproc_dir = model_dir / "preprocessing"
        
        weights_dir.mkdir(parents=True)
        preproc_dir.mkdir(parents=True)
        
        # Copy model weights (flatten structure)
        weight_count = 0
        for fold_dir in sorted(src_strategy.iterdir()):
            if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
                fold_num = fold_dir.name.split("_")[1]
                
                for seed_dir in sorted(fold_dir.iterdir()):
                    if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                        seed_num = seed_dir.name.split("_")[1]
                        
                        for pkl_file in seed_dir.glob("*.pkl"):
                            # Flatten name: fold1_seed42_classifier.pkl
                            new_name = f"fold{fold_num}_seed{seed_num}_{pkl_file.name}"
                            shutil.copy2(pkl_file, weights_dir / new_name)
                            weight_count += 1
        
        print(f"  Copied {weight_count} weight files")
        
        # Copy preprocessing artifacts (same for all models)
        preproc_count = 0
        
        # Common features from root
        common_features = preprocessing_path / "common_features.json"
        if common_features.exists():
            shutil.copy2(common_features, preproc_dir / "common_features.json")
            preproc_count += 1
        
        # Fold artifacts (use fold 5)
        fold_dir = preprocessing_path / "fold_5"
        if fold_dir.exists():
            for f in fold_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, preproc_dir / f.name)
                    preproc_count += 1
        
        print(f"  Copied {preproc_count} preprocessing files")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DEPLOY DIRECTORY READY")
    print("=" * 60)
    
    print(f"\nLocation: {deploy_path}")
    print("\nStructure:")
    
    for item in sorted(deploy_path.iterdir()):
        if item.is_dir():
            print(f"  {item.name}/")
            for sub in sorted(item.iterdir()):
                if sub.is_dir():
                    file_count = len(list(sub.iterdir()))
                    print(f"    {sub.name}/ ({file_count} files)")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in deploy_path.rglob("*") if f.is_file())
    print(f"\nTotal size: {total_size / (1024 * 1024):.2f} MB")
    
    print("\n" + "-" * 60)
    print("NEXT STEPS:")
    print("-" * 60)
    print("1. Open File Explorer and navigate to:")
    print(f"   {deploy_path}")
    print("2. Select all 3 model folders")
    print("3. Drag them to your Google Drive models folder")
    print("-" * 60)


if __name__ == "__main__":
    main()

