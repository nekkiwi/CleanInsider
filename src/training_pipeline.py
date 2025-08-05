# file: src/training_pipeline.py

import itertools
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from tqdm import tqdm

from .training.training_helpers import (
    calculate_spread_cost,
    evaluate_fold,
    find_optimal_threshold,
    save_strategy_results,
    select_features_for_fold,
)


class ModelTrainer:
    def __init__(self, num_folds: int):
        self.num_folds = num_folds
        project_root = Path(__file__).resolve().parent.parent
        self.features_base_path = project_root / "data/scrapers/features"
        self.targets_base_path = project_root / "data/scrapers/targets"
        self.output_dir = project_root / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_data_for_set(self, feature_path: Path, label_path: Path):
        """Loads and merges a feature set with its corresponding labels."""
        if not feature_path.exists() or not label_path.exists():
            return None
        features_df, labels_df = pd.read_parquet(feature_path), pd.read_parquet(
            label_path
        )
        features_df["Filing Date"] = pd.to_datetime(features_df["Filing Date"])
        labels_df["Filing Date"] = pd.to_datetime(labels_df["Filing Date"])
        return pd.merge(
            features_df, labels_df, on=["Ticker", "Filing Date"], how="inner"
        )

    def _prepare_strategy_data(self, data_df, target_col, threshold_pct):
        """Prepares X and y dataframes for a specific strategy."""
        if data_df is None or target_col not in data_df.columns:
            return None, None, None
        data_df = data_df.dropna(subset=[target_col]).copy()
        if data_df.empty:
            return None, None, None
        y_continuous, y_binary = data_df[target_col], (
            data_df[target_col] >= (threshold_pct / 100.0)
        ).astype(int)
        feature_cols = [
            c
            for c in data_df.columns
            if c not in ["Ticker", "Filing Date"] and not c.startswith("alpha_")
        ]
        return data_df[feature_cols], y_binary, y_continuous

    def _train_models(self, X_tr, y_bin_tr, y_cont_tr, seed):
        """Trains classifier and regressor models."""
        params = {
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
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
        test_labels_path = self.targets_base_path / "test_labels.parquet"
        test_df = self._load_data_for_set(test_features_path, test_labels_path)

        combinations = list(itertools.product(seeds, strategies, binary_thresholds_pct))
        for seed, strategy, bin_thresh in tqdm(
            combinations, desc="Processing Strategies"
        ):
            timepoint, tp, sl = strategy
            target_col = f"alpha_{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"

            # Walk-Forward Validation and Testing Loop
            for fold in range(1, self.num_folds + 1):
                train_features_path = (
                    self.features_base_path / f"fold_{fold}/training_data.parquet"
                )
                train_labels_path = (
                    self.targets_base_path / f"fold_{fold}/training_labels.parquet"
                )
                val_features_path = (
                    self.features_base_path / f"fold_{fold}/validation_data.parquet"
                )
                val_labels_path = (
                    self.targets_base_path / f"fold_{fold}/validation_labels.parquet"
                )

                train_df = self._load_data_for_set(
                    train_features_path, train_labels_path
                )
                val_df = self._load_data_for_set(val_features_path, val_labels_path)
                if train_df is None or val_df is None:
                    continue

                X_tr, y_bin_tr, y_cont_tr = self._prepare_strategy_data(
                    train_df, target_col, bin_thresh
                )
                X_val, y_bin_val, y_cont_val = self._prepare_strategy_data(
                    val_df, target_col, bin_thresh
                )
                if X_tr is None or X_val is None:
                    continue

                selected_features = select_features_for_fold(
                    X_tr, y_bin_tr, top_n, seed
                )
                if not selected_features:
                    continue

                imputation_values = X_tr[selected_features].median()
                X_tr_sel = X_tr[selected_features].fillna(imputation_values)
                X_val_sel = X_val[selected_features].fillna(imputation_values)

                classifier, regressor = self._train_models(
                    X_tr_sel, y_bin_tr, y_cont_tr, seed
                )
                if regressor is None:
                    continue

                # --- EVALUATION 1: On this fold's specific validation set ---
                costs_val = calculate_spread_cost(X_val)
                val_buy_signals = classifier.predict(X_val_sel)
                val_pos_idx = X_val_sel.index[val_buy_signals == 1]

                if not val_pos_idx.empty:
                    val_pred_returns = pd.Series(
                        regressor.predict(X_val_sel.loc[val_pos_idx]), index=val_pos_idx
                    )
                    opt_results = find_optimal_threshold(
                        val_pred_returns,
                        y_cont_val.loc[val_pos_idx],
                        costs_val.loc[val_pos_idx],
                    )
                    val_metrics = evaluate_fold(
                        classifier,
                        regressor,
                        opt_results["optimal_threshold"],
                        X_val_sel,
                        y_bin_val,
                        y_cont_val,
                        costs_val,
                    )
                    if val_metrics:
                        all_validation_results.append(
                            {
                                "Timepoint": timepoint,
                                "TP": tp,
                                "SL": sl,
                                "Threshold": bin_thresh,
                                "Seed": seed,
                                "Fold": fold,
                                **val_metrics,
                            }
                        )

                # --- EVALUATION 2: On the single, static final test set ---
                if test_df is not None:
                    X_test, y_bin_test, y_cont_test = self._prepare_strategy_data(
                        test_df, target_col, bin_thresh
                    )
                    if X_test is not None:
                        X_test_sel = X_test[selected_features].fillna(imputation_values)
                        costs_test = calculate_spread_cost(X_test)
                        # Use the optimal threshold found on THIS fold's validation set
                        test_metrics = evaluate_fold(
                            classifier,
                            regressor,
                            opt_results.get("optimal_threshold"),
                            X_test_sel,
                            y_bin_test,
                            y_cont_test,
                            costs_test,
                        )
                        if test_metrics:
                            all_test_results.append(
                                {
                                    "Timepoint": timepoint,
                                    "TP": tp,
                                    "SL": sl,
                                    "Threshold": bin_thresh,
                                    "Seed": seed,
                                    "Fold": fold,
                                    **test_metrics,
                                }
                            )

        # Save results to two separate files
        if all_validation_results:
            save_strategy_results(
                pd.DataFrame(all_validation_results),
                self.output_dir,
                f"{model_type}_Validation_Metrics",
            )
        if all_test_results:
            save_strategy_results(
                pd.DataFrame(all_test_results),
                self.output_dir,
                f"{model_type}_Test_Metrics",
            )
