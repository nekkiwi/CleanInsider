# file: src/training_pipeline.py

import pandas as pd
from tqdm import tqdm
import itertools
from pathlib import Path
import joblib
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np

from .training.training_helpers import (
    evaluate_fold, save_strategy_results,
    select_features_for_fold
)
from .preprocess.utils import load_preprocessing_artifacts

class ModelTrainer:
    def __init__(self, num_folds: int):
        self.num_folds = num_folds
        project_root = Path(__file__).resolve().parent.parent
        self.features_base_path = project_root / "data/scrapers/features"
        self.targets_base_path = project_root / "data/scrapers/targets"
        self.output_dir = project_root / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_base_path = project_root / "data/models"  # NEW: Base path for models
        self.models_base_path.mkdir(parents=True, exist_ok=True)
        self.preprocessing_artifacts_path = self.features_base_path / "preprocessing"
        
    
        
    def _get_strategy_string(self, strategy):
        """Convert strategy tuple to folder-safe string."""
        timepoint, tp, sl = strategy
        return f"{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"

    def _save_models(self, classifier, regressor, strategy, fold, seed, selected_features, imputation_values):
        """Save trained models and metadata to the specified directory structure."""
        strategy_str = self._get_strategy_string(strategy)
        
        # Create directory structure: data/models/{strategy}/fold_x/seed_x/
        model_dir = self.models_base_path / strategy_str / f"fold_{fold}" / f"seed_{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        classifier_path = model_dir / "classifier.pkl"
        joblib.dump(classifier, classifier_path)
        
        # Save regressor (if it exists)
        if regressor is not None:
            regressor_path = model_dir / "regressor.pkl"
            joblib.dump(regressor, regressor_path)
        
        # Save metadata (features, imputation values)
        metadata = {
            'selected_features': selected_features,
            'imputation_values': imputation_values.to_dict(),
            'strategy': strategy,
            'fold': fold,
            'seed': seed
        }
        metadata_path = model_dir / "metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        print(f"[MODEL-SAVE] Saved models for {strategy_str}/fold_{fold}/seed_{seed}")
        return model_dir

    def _load_data_for_set(self, feature_path: Path, label_path: Path):
        """Loads and merges a feature set with its corresponding labels and spread data."""
        if not feature_path.exists() or not label_path.exists(): return None
        
        # Determine spread file name
        if "training" in feature_path.name:
            spread_filename = "training_spreads.parquet"
        elif "validation" in feature_path.name:
            spread_filename = "validation_spreads.parquet"
        elif "test" in feature_path.name:
            spread_filename = "test_spreads.parquet"
        else:
            spread_filename = None

        features_df = pd.read_parquet(feature_path)
        labels_df = pd.read_parquet(label_path)
        
        features_df['Filing Date'] = pd.to_datetime(features_df['Filing Date'])
        labels_df['Filing Date'] = pd.to_datetime(labels_df['Filing Date'])
        
        merged_df = pd.merge(features_df, labels_df, on=['Ticker', 'Filing Date'], how='inner')

        if spread_filename:
            spread_path = label_path.parent / spread_filename
            if spread_path.exists():
                spread_df = pd.read_parquet(spread_path)
                spread_df['Filing Date'] = pd.to_datetime(spread_df['Filing Date'])
                merged_df = pd.merge(merged_df, spread_df, on=['Ticker', 'Filing Date'], how='left')
            else:
                print(f"  [WARN] Spread file not found, but expected: {spread_path}")
                merged_df['corwin_schultz_spread'] = np.nan
        
        return merged_df

    def _prepare_strategy_data(self, data_df, target_col, threshold_pct):
        """Prepares X and y dataframes for a specific strategy."""
        if data_df is None or target_col not in data_df.columns: return None, None, None
        data_df = data_df.dropna(subset=[target_col]).copy()
        if data_df.empty: return None, None, None
        y_continuous, y_binary = data_df[target_col], (data_df[target_col] >= (threshold_pct / 100.0)).astype(int)
        feature_cols = [c for c in data_df.columns if c not in ['Ticker', 'Filing Date'] and not c.startswith('alpha_')]
        return data_df[feature_cols], y_binary, y_continuous

    def _train_models(self, X_tr, y_bin_tr, y_cont_tr, seed):
        """Trains classifier and regressor models."""
        params = {'random_state': seed, 'n_jobs': -1, 'verbosity': -1, 'subsample': 0.8, 'colsample_bytree': 0.8}
        classifier, regressor = LGBMClassifier(**params), LGBMRegressor(**params)
        classifier.fit(X_tr, y_bin_tr)
        pos_idx = y_bin_tr[y_bin_tr == 1].index
        if not pos_idx.empty:
            regressor.fit(X_tr.loc[pos_idx], y_cont_tr.loc[pos_idx])
            return classifier, regressor
        return classifier, None

    def run(self, strategies, binary_thresholds_pct, model_type, top_n, seeds):
        print(f"\n### STARTING: {model_type} Walk-Forward Training & Final Test ###")
        all_validation_results, all_test_results = [], []

        # Load the single, static test set ONCE at the beginning.
        test_features_path = self.features_base_path / "test_set" / "test_data.parquet"
        test_labels_path = self.targets_base_path / "test_set" / "test_labels.parquet"
        test_df = self._load_data_for_set(test_features_path, test_labels_path)

        combinations = list(itertools.product(seeds, strategies, binary_thresholds_pct))
        for seed, strategy, bin_thresh in tqdm(combinations, desc="Processing Strategies"):
            timepoint, tp, sl = strategy
            target_col = f"alpha_{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"

            # Walk-Forward Validation and Testing Loop
            for fold in range(1, self.num_folds + 1):
                train_features_path = self.features_base_path / f"fold_{fold}/training_data.parquet"
                train_labels_path = self.targets_base_path / f"fold_{fold}/training_labels.parquet"
                val_features_path = self.features_base_path / f"fold_{fold}/validation_data.parquet"
                val_labels_path = self.targets_base_path / f"fold_{fold}/validation_labels.parquet"
                
                train_df = self._load_data_for_set(train_features_path, train_labels_path)
                val_df = self._load_data_for_set(val_features_path, val_labels_path)
                if train_df is None or val_df is None: continue

                X_tr, y_bin_tr, y_cont_tr = self._prepare_strategy_data(train_df, target_col, bin_thresh)
                X_val, y_bin_val, y_cont_val = self._prepare_strategy_data(val_df, target_col, bin_thresh)
                if X_tr is None or X_val is None: continue

                # --- IMPUTATION & FEATURE SELECTION (Corrected Workflow) ---
                # 1. Learn imputation values from the training set only
                imputation_values = X_tr.median()
                X_tr_imputed = X_tr.fillna(imputation_values)

                # 2. Select features based on the properly imputed training data
                selected_features = select_features_for_fold(X_tr_imputed, y_bin_tr, top_n, seed)
                if not selected_features: continue
                
                # 3. Create final model inputs with selected features
                X_tr_sel = X_tr_imputed[selected_features]
                X_val_sel = X_val[selected_features].fillna(imputation_values) # Apply same imputation to validation set

                classifier, regressor = self._train_models(X_tr_sel, y_bin_tr, y_cont_tr, seed)
                if regressor is None: continue

                # --- COST CALCULATION: Prioritize Corwin-Schultz, fallback to estimation ---
                if 'corwin_schultz_spread' in val_df.columns:
                    costs_val = val_df['corwin_schultz_spread'].reindex(X_val.index).fillna(0.0005)
                else:
                    print("[WARN] 'corwin_schultz_spread' not found in validation set. Defaulting to 5bps cost.")
                    costs_val = pd.Series(0.0005, index=X_val.index)
                
                val_metrics = evaluate_fold(classifier, regressor, X_val_sel, y_bin_val, y_cont_val, costs_val)
                if val_metrics:
                    all_validation_results.append({'Timepoint': timepoint, 'TP': tp, 'SL': sl, 'Threshold': bin_thresh, 'Seed': seed, 'Fold': fold, **val_metrics})

                self._save_models(
                    classifier=classifier,
                    regressor=regressor,
                    strategy=strategy,
                    fold=fold,
                    seed=seed,
                    selected_features=selected_features,
                    imputation_values=imputation_values
                )

                # --- EVALUATION 2: On the single, static final test set ---
                if test_df is not None:
                    X_test, y_bin_test, y_cont_test = self._prepare_strategy_data(test_df, target_col, bin_thresh)
                    if X_test is not None:
                        # Apply the imputation learned from the training set
                        X_test_sel = X_test[selected_features].fillna(imputation_values)
                        
                        if 'corwin_schultz_spread' in test_df.columns:
                            costs_test = test_df['corwin_schultz_spread'].reindex(X_test.index).fillna(0.0005)
                        else:
                            print("[WARN] 'corwin_schultz_spread' not found in test set. Defaulting to 5bps cost.")
                            costs_test = pd.Series(0.0005, index=X_test.index)
                            
                        test_metrics = evaluate_fold(classifier, regressor, X_test_sel, y_bin_test, y_cont_test, costs_test)
                        if test_metrics:
                            all_test_results.append({'Timepoint': timepoint, 'TP': tp, 'SL': sl, 'Threshold': bin_thresh, 'Seed': seed, 'Fold': fold, **test_metrics})

        # Save results to two separate files
        if all_validation_results:
            save_strategy_results(pd.DataFrame(all_validation_results), self.output_dir, f"{model_type}_Validation_Metrics")
        if all_test_results:
            save_strategy_results(pd.DataFrame(all_test_results), self.output_dir, f"{model_type}_Test_Metrics")

    def load_model(self, strategy, fold, seed, model_type='both'):
            """Load a saved model for inference."""
            strategy_str = self._get_strategy_string(strategy)
            model_dir = self.models_base_path / strategy_str / f"fold_{fold}" / f"seed_{seed}"
            
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            # Load metadata
            metadata_path = model_dir / "metadata.pkl"
            metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
            
            # Load models
            if model_type in ['classifier', 'both']:
                classifier_path = model_dir / "classifier.pkl"
                classifier = joblib.load(classifier_path) if classifier_path.exists() else None
            else:
                classifier = None
                
            if model_type in ['regressor', 'both']:
                regressor_path = model_dir / "regressor.pkl"
                regressor = joblib.load(regressor_path) if regressor_path.exists() else None
            else:
                regressor = None
            
            return classifier, regressor, metadata