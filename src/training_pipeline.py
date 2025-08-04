# file: src/training_pipeline.py

import itertools
import os

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from tqdm import tqdm

from .training.training_helpers import (
    calculate_spread_cost,
    evaluate_test_fold,
    find_optimal_threshold,
    save_strategy_results,
    select_features_for_fold,
)


class ModelTrainer:
    """
    Orchestrates a simplified walk-forward training and evaluation pipeline,
    focusing on feature selection and modeling.
    """

    def __init__(self, n_splits: int):
        self.n_splits = n_splits
        base_data_path = os.path.join(os.path.dirname(__file__), "..", "data")
        self.features_base_path = os.path.join(base_data_path, "scrapers", "features")
        self.targets_base_path = os.path.join(base_data_path, "scrapers", "targets")
        self.output_stats_dir = os.path.join(base_data_path, "training_stats")
        os.makedirs(self.output_stats_dir, exist_ok=True)

    def _load_fold_data(self, fold_id: str or int):
        """Loads and merges features and targets for a specific fold."""
        features_path = os.path.join(
            self.features_base_path, f"fold_{fold_id}", "preprocessed_fold.parquet"
        )
        targets_path = os.path.join(
            self.targets_base_path, f"fold_{fold_id}", "targets.parquet"
        )

        if not os.path.exists(features_path) or not os.path.exists(targets_path):
            print(f"[WARN] Data for fold '{fold_id}' not found. Skipping.")
            return None

        features_df = pd.read_parquet(features_path)
        targets_df = pd.read_parquet(targets_path)
        features_df["Filing Date"] = pd.to_datetime(features_df["Filing Date"])
        targets_df["Filing Date"] = pd.to_datetime(targets_df["Filing Date"])

        return (
            pd.merge(features_df, targets_df, on=["Ticker", "Filing Date"], how="inner")
            .sort_values(by="Filing Date")
            .reset_index(drop=True)
        )

    def _prepare_strategy_data(self, data_df, target_col, threshold_pct):
        """Prepares X and y dataframes for a specific strategy."""
        if target_col not in data_df.columns:
            return None, None, None
        data_df = data_df.dropna(subset=[target_col]).copy()
        if data_df.empty:
            return None, None, None

        y_continuous = data_df[target_col]
        y_binary = (y_continuous >= (threshold_pct / 100.0)).astype(int)
        y_continuous.name, y_binary.name = target_col, f"{target_col}_binary"

        feature_cols = [
            col
            for col in data_df.columns
            if col not in ["Ticker", "Filing Date"] and not col.startswith("alpha_")
        ]
        X = data_df[feature_cols]
        return X, y_binary, y_continuous

    def _train_models(self, X_tr, y_bin_tr, y_cont_tr, seed, model_type):
        """Trains classifier and regressor models with default parameters."""
        lgbm_params = {
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        if model_type == "LightGBM":
            classifier = LGBMClassifier(**lgbm_params)
            regressor = LGBMRegressor(**lgbm_params)

        classifier.fit(X_tr, y_bin_tr)

        pos_train_idx = y_bin_tr[y_bin_tr == 1].index
        if not pos_train_idx.empty:
            regressor.fit(X_tr.loc[pos_train_idx], y_cont_tr.loc[pos_train_idx])
            return classifier, regressor
        return classifier, None

    def run(self, strategies, binary_thresholds_pct, model_type, top_n, seeds):
        print(f"\n### STARTING: Simplified {model_type} Walk-Forward ###")
        all_results = []

        combinations = list(itertools.product(seeds, strategies, binary_thresholds_pct))
        for seed, strategy, bin_thresh in tqdm(
            combinations, desc="Processing Strategies"
        ):
            timepoint, tp, sl = strategy
            target_col = f"alpha_{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"

            # Walk-forward loop: Train on Fold N, Validate/Test on Fold N+1
            for fold in range(1, self.n_splits):
                train_df = self._load_fold_data(fold)
                test_df = self._load_fold_data(fold + 1)
                if train_df is None or test_df is None:
                    continue

                # 1. Prepare data
                X_train_val, y_bin_train_val, y_cont_train_val = (
                    self._prepare_strategy_data(train_df, target_col, bin_thresh)
                )
                X_test, y_bin_test, y_cont_test = self._prepare_strategy_data(
                    test_df, target_col, bin_thresh
                )
                if X_train_val is None or X_test is None:
                    continue

                costs_val = calculate_spread_cost(X_train_val)
                costs_test = calculate_spread_cost(X_test)

                val_size = int(len(X_train_val) * 0.2)
                train_indices, val_indices = (
                    X_train_val.index[:-val_size],
                    X_train_val.index[-val_size:],
                )
                X_tr, y_bin_tr, y_cont_tr = (
                    X_train_val.loc[train_indices],
                    y_bin_train_val.loc[train_indices],
                    y_cont_train_val.loc[train_indices],
                )
                X_val, y_cont_val = (
                    X_train_val.loc[val_indices],
                    y_cont_train_val.loc[val_indices],
                )
                costs_val_split = costs_val.loc[val_indices]

                selected_features = select_features_for_fold(
                    X_tr, y_bin_tr, top_n, seed
                )
                if not selected_features:
                    continue

                X_tr_sel = X_tr[selected_features].fillna(
                    X_tr[selected_features].median()
                )
                X_val_sel = X_val[selected_features].fillna(
                    X_tr[selected_features].median()
                )
                X_test_sel = X_test[selected_features].fillna(
                    X_tr[selected_features].median()
                )

                classifier, regressor = self._train_models(
                    X_tr_sel, y_bin_tr, y_cont_tr, seed, model_type
                )
                if regressor is None:
                    continue

                val_buy_signals = classifier.predict(X_val_sel)
                val_pos_idx = X_val_sel.index[val_buy_signals == 1]
                if val_pos_idx.empty:
                    continue
                val_predicted_returns = pd.Series(
                    regressor.predict(X_val_sel.loc[val_pos_idx]), index=val_pos_idx
                )

                # Pass costs to the optimization and evaluation functions
                opt_results = find_optimal_threshold(
                    val_predicted_returns,
                    y_cont_val.loc[val_pos_idx],
                    costs_val_split.loc[val_pos_idx],
                )

                fold_metrics = evaluate_test_fold(
                    classifier,
                    regressor,
                    opt_results["optimal_threshold"],
                    X_test_sel,
                    y_bin_test,
                    y_cont_test,
                    costs_test,
                )

                if fold_metrics:
                    result_row = {
                        "Timepoint": timepoint,
                        "TP": tp,
                        "SL": sl,
                        "Binary Threshold": f"{bin_thresh}%",
                        "Seed": seed,
                        "Fold": fold,
                        "Model": model_type,
                    }
                    result_row.update(fold_metrics)
                    all_results.append(result_row)

        if all_results:
            results_df = pd.DataFrame(all_results)
            save_strategy_results(
                results_df, self.output_stats_dir, model_type, "alpha"
            )
