# file: src/training_pipeline.py

import pandas as pd
from tqdm import tqdm
import itertools
from pathlib import Path
from lightgbm import LGBMClassifier, LGBMRegressor
from .training.training_helpers import (
    find_optimal_threshold,
    evaluate_test_fold,
    save_strategy_results,
    select_features_for_fold,
    calculate_spread_cost
)

class ModelTrainer:
    def __init__(self, num_folds: int):
        self.num_folds = num_folds
        project_root = Path(__file__).resolve().parent.parent
        self.features_base_path = project_root / "data/scrapers/features"
        self.targets_base_path = project_root / "data/scrapers/targets"
        self.output_dir = project_root / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_and_merge_data(self, feature_path: Path, targets_df: pd.DataFrame):
        """Helper to load features and merge with a pre-loaded targets DataFrame."""
        if not feature_path.exists() or targets_df is None: return None
        features_df = pd.read_parquet(feature_path)
        features_df['Filing Date'] = pd.to_datetime(features_df['Filing Date'])
        return pd.merge(features_df, targets_df, on=['Ticker', 'Filing Date'], how='inner')

    def _prepare_strategy_data(self, data_df: pd.DataFrame, target_col: str, threshold_pct: int):
        """Prepares X and y dataframes for a specific strategy."""
        if data_df is None or target_col not in data_df.columns: return None, None, None
        data_df = data_df.dropna(subset=[target_col]).copy()
        if data_df.empty: return None, None, None
        y_continuous = data_df[target_col]
        y_binary = (y_continuous >= (threshold_pct / 100.0)).astype(int)
        feature_cols = [col for col in data_df.columns if col not in ['Ticker', 'Filing Date'] and not col.startswith('alpha_')]
        X = data_df[feature_cols]
        return X, y_binary, y_continuous

    def _train_models(self, X_tr, y_bin_tr, y_cont_tr, seed, model_type):
        """Trains classifier and regressor models."""
        lgbm_params = {'random_state': seed, 'n_jobs': -1, 'verbosity': -1, 'subsample': 0.8, 'colsample_bytree': 0.8}
        classifier = LGBMClassifier(**lgbm_params)
        regressor = LGBMRegressor(**lgbm_params)
        classifier.fit(X_tr, y_bin_tr)
        pos_train_idx = y_bin_tr[y_bin_tr == 1].index
        if not pos_train_idx.empty:
            regressor.fit(X_tr.loc[pos_train_idx], y_cont_tr.loc[pos_train_idx])
            return classifier, regressor
        return classifier, None

    def run(self, strategies, binary_thresholds_pct, model_type, top_n, seeds):
        print(f"\n### STARTING: {model_type} Walk-Forward Training & Final Test ###")
        
        all_validation_results, all_test_results = [], []
        
        # Load all targets once to avoid repeated file I/O
        all_targets_df = pd.read_parquet(self.targets_base_path / "master_targets.parquet")
        all_targets_df['Filing Date'] = pd.to_datetime(all_targets_df['Filing Date'])
        
        # Load the static, final test set once
        test_feature_path = self.features_base_path / "111final_test_set_unprocessed.parquet" # Explicit path for the test set
        final_test_df = self._load_and_merge_data(test_feature_path, all_targets_df)
        if final_test_df is None:
            print("[WARN] Final test set not found. Skipping final test evaluation.")

        combinations = list(itertools.product(seeds, strategies, binary_thresholds_pct))
        for seed, strategy, bin_thresh in tqdm(combinations, desc="Processing Strategies"):
            timepoint, tp, sl = strategy
            target_col = f"alpha_{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"
            
            final_model_artifacts = {}

            # Walk-Forward Validation Loop using the instance's num_folds
            for fold in range(1, self.num_folds + 1):
                train_feature_path = self.features_base_path / f"fold_{fold}/training_data.parquet"
                val_feature_path = self.features_base_path / f"fold_{fold}/evaluation_data.parquet"
                
                train_df = self._load_and_merge_data(train_feature_path, all_targets_df)
                val_df = self._load_and_merge_data(val_feature_path, all_targets_df)

                if train_df is None or val_df is None or train_df.empty or val_df.empty: continue

                X_tr, y_bin_tr, y_cont_tr = self._prepare_strategy_data(train_df, target_col, bin_thresh)
                X_val, y_bin_val, y_cont_val = self._prepare_strategy_data(val_df, target_col, bin_thresh)

                if X_tr is None or X_val is None or X_tr.empty or X_val.empty: continue

                selected_features = select_features_for_fold(X_tr, y_bin_tr, top_n, seed)
                if not selected_features: continue

                X_tr_sel = X_tr[selected_features].fillna(X_tr[selected_features].median())
                X_val_sel = X_val[selected_features].fillna(X_tr[selected_features].median())
                
                classifier, regressor = self._train_models(X_tr_sel, y_bin_tr, y_cont_tr, seed, model_type)
                if regressor is None: continue
                
                costs_val = calculate_spread_cost(X_val)
                val_buy_signals = classifier.predict(X_val_sel)
                val_pos_idx = X_val_sel.index[val_buy_signals == 1]
                
                if val_pos_idx.empty: continue
                
                val_pred_returns = pd.Series(regressor.predict(X_val_sel.loc[val_pos_idx]), index=val_pos_idx)
                opt_results = find_optimal_threshold(val_pred_returns, y_cont_val.loc[val_pos_idx], costs_val.loc[val_pos_idx])
                
                val_metrics = evaluate_test_fold(classifier, regressor, opt_results.get('optimal_threshold'), X_val_sel, y_bin_val, y_cont_val, costs_val)
                if val_metrics:
                    # --- THIS IS THE FIX ---
                    result_row = {
                        'Timepoint': timepoint, 'TP': tp, 'SL': sl, 'Threshold': bin_thresh, 
                        'Seed': seed, 'Fold': fold, **val_metrics
                    }
                    all_validation_results.append(result_row)
                
                if fold == self.num_folds:
                    final_model_artifacts = {'classifier': classifier, 'regressor': regressor, 'selected_features': selected_features, 'imputation_values': X_tr[selected_features].median(), 'optimal_threshold': opt_results.get('optimal_threshold')}

            # Final Test Evaluation
            if final_test_df is not None and final_model_artifacts:
                X_test, y_bin_test, y_cont_test = self._prepare_strategy_data(final_test_df, target_col, bin_thresh)
                if X_test is not None and not X_test.empty:
                    X_test_sel = X_test[final_model_artifacts['selected_features']].fillna(final_model_artifacts['imputation_values'])
                    costs_test = calculate_spread_cost(X_test)
                    
                    test_metrics = evaluate_test_fold(final_model_artifacts['classifier'], final_model_artifacts['regressor'], final_model_artifacts['optimal_threshold'], X_test_sel, y_bin_test, y_cont_test, costs_test)
                    if test_metrics:
                        test_row = {
                            'Timepoint': timepoint, 'TP': tp, 'SL': sl, 'Threshold': bin_thresh, 
                            'Seed': seed, **test_metrics
                        }
                        all_test_results.append(test_row)

        # Save Results to Excel
        if all_validation_results:
            save_strategy_results(pd.DataFrame(all_validation_results), self.output_dir, f"{model_type}_Validation_Metrics", "alpha_validation")
        if all_test_results:
            save_strategy_results(pd.DataFrame(all_test_results), self.output_dir, f"{model_type}_Test_Metrics", "alpha_test")
