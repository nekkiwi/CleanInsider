# file: src/training_pipeline.py

import pandas as pd
from tqdm import tqdm
import itertools
from pathlib import Path
import joblib
from lightgbm import LGBMClassifier, LGBMRegressor
import optuna
import numpy as np

from .training.training_helpers import (
    evaluate_fold, evaluate_classifier_only, save_strategy_results,
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
        self.best_hyperparams = None
        # Flags to control features
        self.enable_separate_tuning = False
        self.enable_min_signal_gate = True
        self.min_signal_gate_threshold = 0.5
        # Optional: separate best params
        self.best_hyperparams_classifier = None
        self.best_hyperparams_regressor = None
        # NEW: toggle regressor usage
        self.use_regressor = False
        
    
        
    def _get_strategy_string(self, strategy):
        """Convert strategy tuple to folder-safe string."""
        timepoint, tp, sl = strategy
        return f"{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"

    def _save_models(self, classifier, regressor, strategy, fold, seed, selected_features, imputation_values, threshold_pct: int | float):
        """Save trained models and metadata to the specified directory structure."""
        strategy_str = self._get_strategy_string(strategy)
        
        # Create directory structure: data/models/{strategy}/thr_{threshold}/fold_x/seed_x/
        thr_str = str(threshold_pct).replace('.', 'p')
        model_dir = self.models_base_path / strategy_str / f"thr_{thr_str}" / f"fold_{fold}" / f"seed_{seed}"
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
            'seed': seed,
            'threshold_pct': threshold_pct,
        }
        metadata_path = model_dir / "metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        print(f"[MODEL-SAVE] Saved models for {strategy_str}/fold_{fold}/seed_{seed}")
        return model_dir

    def _load_data_for_set(self, feature_path: Path, label_path: Path):
        """Loads and merges a feature set with its corresponding labels."""
        if not feature_path.exists() or not label_path.exists(): return None

        features_df = pd.read_parquet(feature_path)
        labels_df = pd.read_parquet(label_path)
        
        features_df['Filing Date'] = pd.to_datetime(features_df['Filing Date'])
        labels_df['Filing Date'] = pd.to_datetime(labels_df['Filing Date'])
        
        merged_df = pd.merge(features_df, labels_df, on=['Ticker', 'Filing Date'], how='inner')

        return merged_df

    def _prepare_strategy_data(self, data_df, target_col, threshold_pct):
        """Prepares X and y dataframes for a specific strategy."""
        if data_df is None or target_col not in data_df.columns: return None, None, None
        data_df = data_df.dropna(subset=[target_col]).copy()
        if data_df.empty: return None, None, None
        y_continuous, y_binary = data_df[target_col], (data_df[target_col] >= (threshold_pct / 100.0)).astype(int)
        # Exclude non-feature identifiers
        feature_cols = [
            c for c in data_df.columns
            if c not in ['Ticker', 'Filing Date', 'split_id'] and not c.startswith('alpha_')
        ]
        return data_df[feature_cols], y_binary, y_continuous

    def _train_models(self, X_tr, y_bin_tr, y_cont_tr, seed):
        """Trains classifier and regressor models."""
        params_cls = {
            'random_state': seed,
            'n_jobs': -1,
            'verbosity': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'num_leaves': 15,
            'min_child_samples': 150,
            'reg_alpha': 0.5,
            'reg_lambda': 0.8,
        }
        params_reg = dict(params_cls)
        # If a unified tuning was performed previously
        if self.best_hyperparams and not self.enable_separate_tuning:
            for k, v in self.best_hyperparams.items():
                if k != 'random_state':
                    params_cls[k] = v
                    params_reg[k] = v
        # If separate tuning provided
        if self.enable_separate_tuning:
            if self.best_hyperparams_classifier:
                for k, v in self.best_hyperparams_classifier.items():
                    if k != 'random_state':
                        params_cls[k] = v
            if self.best_hyperparams_regressor:
                for k, v in self.best_hyperparams_regressor.items():
                    if k != 'random_state':
                        params_reg[k] = v

        classifier, regressor = LGBMClassifier(**params_cls), LGBMRegressor(**params_reg)
        classifier.fit(X_tr, y_bin_tr)
        pos_idx = y_bin_tr[y_bin_tr == 1].index
        if not pos_idx.empty:
            regressor.fit(X_tr.loc[pos_idx], y_cont_tr.loc[pos_idx])
            return classifier, regressor
        return classifier, None

    def _format_hparam_suffix(self, params: dict) -> str:
        """Create a short, filename-safe suffix summarizing tuned hyperparameters."""
        if not params:
            return "default"
        def fmt_float(x, nd=2):
            return ("%0." + str(nd) + "f") % float(x)
        parts = [
            f"nl{int(params.get('num_leaves', 0))}",
            f"mcs{int(params.get('min_child_samples', 0))}",
            f"lr{fmt_float(params.get('learning_rate', 0.0))}",
            f"ne{int(params.get('n_estimators', 0))}",
            f"ss{fmt_float(params.get('subsample', 0.0))}",
            f"cs{fmt_float(params.get('colsample_bytree', 0.0))}",
            f"ra{fmt_float(params.get('reg_alpha', 0.0))}",
            f"rl{fmt_float(params.get('reg_lambda', 0.0))}",
        ]
        return "_" + "-".join(parts)

    def _tune_hyperparameters(self, strategies, binary_thresholds_pct, top_n, seed_for_tuning=42, n_trials=40, timeout_seconds=None, folds_mode='all', specific_folds=None):
        """
        Tune LightGBM hyperparameters using Optuna on a single representative fold (fold 1)
        and a representative strategy/threshold, to avoid data leakage across the full WFO.
        The tuned params are then applied globally for the run.
        """
        if not strategies or not binary_thresholds_pct:
            return None
        # Representative combo: first strategy and a central/1.0 threshold if available
        strategy = strategies[0]
        thresholds_sorted = sorted(binary_thresholds_pct)
        threshold = 1 if 1 in thresholds_sorted else thresholds_sorted[len(thresholds_sorted)//2]
        timepoint, tp, sl = strategy
        target_col = f"alpha_{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"

        # Choose folds for tuning: all validation folds, or last only, or a provided list
        if specific_folds is not None and len(specific_folds) > 0:
            folds = list(specific_folds)
        elif folds_mode == 'last':
            folds = [self.num_folds]
        else:
            folds = list(range(1, self.num_folds + 1))

        # --- Pre-scan folds and thresholds to ensure we tune on viable data (non-empty positives) ---
        # Build prepared datasets per fold for a chosen threshold. Prefer threshold 1, otherwise pick the one that yields the most positives across folds.
        def prepare_fold_data(fold_idx: int, thr: float):
            train_features_path = self.features_base_path / f"fold_{fold_idx}/training_data.parquet"
            train_labels_path = self.targets_base_path / f"fold_{fold_idx}/training_labels.parquet"
            val_features_path = self.features_base_path / f"fold_{fold_idx}/validation_data.parquet"
            val_labels_path = self.targets_base_path / f"fold_{fold_idx}/validation_labels.parquet"
            train_df = self._load_data_for_set(train_features_path, train_labels_path)
            val_df = self._load_data_for_set(val_features_path, val_labels_path)
            if train_df is None or val_df is None:
                return None
            X_tr, y_bin_tr, y_cont_tr = self._prepare_strategy_data(train_df, target_col, thr)
            X_val, y_bin_val, y_cont_val = self._prepare_strategy_data(val_df, target_col, thr)
            if X_tr is None or X_val is None:
                return None
            # Impute & select features using training only
            imputation_values = X_tr.median()
            X_tr_imputed = X_tr.fillna(imputation_values)
            selected_features = select_features_for_fold(X_tr_imputed, y_bin_tr, top_n, seed_for_tuning)
            if not selected_features:
                return None
            X_tr_sel = X_tr_imputed[selected_features]
            X_val_sel = X_val[selected_features].fillna(imputation_values)
            pos_idx = y_bin_tr[y_bin_tr == 1].index
            return {
                'X_tr_sel': X_tr_sel,
                'y_bin_tr': y_bin_tr,
                'y_cont_tr': y_cont_tr,
                'X_val_sel': X_val_sel,
                'y_bin_val': y_bin_val,
                'y_cont_val': y_cont_val,
                'num_pos': int(len(pos_idx))
            }

        prep_by_fold = {}
        chosen_threshold = None
        # Try preferred threshold first, then fall back to others ordered by proximity to 1
        candidate_thresholds = [threshold] + [t for t in thresholds_sorted if t != threshold]
        for thr in candidate_thresholds:
            prep_by_fold.clear()
            total_pos = 0
            for f in folds:
                prep = prepare_fold_data(f, thr)
                if prep is None:
                    continue
                if prep['num_pos'] > 0:
                    prep_by_fold[f] = prep
                    total_pos += prep['num_pos']
            if len(prep_by_fold) > 0 and total_pos > 0:
                chosen_threshold = thr
                break

        # If nothing viable under chosen folds (e.g., folds_mode='last' and no positives), fall back to all folds
        if chosen_threshold is None and folds_mode == 'last':
            folds = list(range(1, self.num_folds + 1))
            for thr in candidate_thresholds:
                prep_by_fold.clear()
                total_pos = 0
                for f in folds:
                    prep = prepare_fold_data(f, thr)
                    if prep is None:
                        continue
                    if prep['num_pos'] > 0:
                        prep_by_fold[f] = prep
                        total_pos += prep['num_pos']
                if len(prep_by_fold) > 0 and total_pos > 0:
                    chosen_threshold = thr
                    break

        if chosen_threshold is None or len(prep_by_fold) == 0:
            return None

        def objective(trial: optuna.Trial) -> float:
            params = {
                'random_state': seed_for_tuning,
                'n_jobs': -1,
                'verbosity': -1,
                'num_leaves': trial.suggest_int('num_leaves', 16, 64),
                'min_child_samples': trial.suggest_int('min_child_samples', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
            }

            fold_scores = []
            for f, prep in prep_by_fold.items():
                clf = LGBMClassifier(**params)
                clf.fit(prep['X_tr_sel'], prep['y_bin_tr'])
                pos_idx = prep['y_bin_tr'][prep['y_bin_tr'] == 1].index
                if len(pos_idx) == 0:
                    continue
                reg = LGBMRegressor(**params)
                reg.fit(prep['X_tr_sel'].loc[pos_idx], prep['y_cont_tr'].loc[pos_idx])

                metrics = evaluate_fold(clf, reg, prep['X_val_sel'], prep['y_bin_val'], prep['y_cont_val'])
                if not metrics:
                    continue
                score = metrics.get('Adj Sharpe (Net)')
                if pd.isna(score):
                    score = metrics.get('Sharpe (Net)')
                if score is not None:
                    fold_scores.append(float(score))

            if len(fold_scores) == 0:
                return -1e9
            return float(np.nanmean(fold_scores))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)
        best_params = study.best_params if study.best_trial is not None else None
        self.best_hyperparams = best_params
        return best_params

    def run(self, strategies, binary_thresholds_pct, model_type, top_n, seeds, tune_hyperparams=False, tuning_trials=40, tuning_timeout=None, tuning_folds_mode='all'):
        print(f"\n### STARTING: {model_type} Walk-Forward Training & Final Test ###")
        all_validation_results, all_test_results = [], []

        # Load the single, static test set ONCE at the beginning.
        test_features_path = self.features_base_path / "test_set" / "test_data.parquet"
        test_labels_path = self.targets_base_path / "test_set" / "test_labels.parquet"
        test_df = self._load_data_for_set(test_features_path, test_labels_path)

        # Optional: one-off hyperparameter tuning using fold 1 only (no leakage into test)
        if tune_hyperparams:
            print("[TUNING] Starting Optuna tuning across folds (mode: %s) with a representative strategy/threshold..." % tuning_folds_mode)
            tuned = self._tune_hyperparameters(
                strategies,
                binary_thresholds_pct,
                top_n,
                seed_for_tuning=seeds[0],
                n_trials=tuning_trials,
                timeout_seconds=tuning_timeout,
                folds_mode=tuning_folds_mode,
                specific_folds=None,
            )
            if tuned:
                print(f"[TUNING] Best params: {tuned}")
            else:
                print("[TUNING] Tuning skipped or failed; using default params.")

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
                if self.use_regressor and regressor is None: continue

                # No explicit trading-cost model in training; targets are already spread-adjusted
                if self.use_regressor:
                    val_metrics = evaluate_fold(
                        classifier, regressor, X_val_sel, y_bin_val, y_cont_val,
                        use_min_signal_gate=self.enable_min_signal_gate,
                        signal_prob_threshold=self.min_signal_gate_threshold
                    )
                else:
                    val_metrics = evaluate_classifier_only(
                        classifier, X_val_sel, y_bin_val, y_cont_val,
                        use_min_signal_gate=self.enable_min_signal_gate,
                        signal_prob_threshold=self.min_signal_gate_threshold
                    )
                if val_metrics:
                    all_validation_results.append({'Timepoint': timepoint, 'TP': tp, 'SL': sl, 'Threshold': bin_thresh, 'Seed': seed, 'Fold': fold, **val_metrics})

                self._save_models(
                    classifier=classifier,
                    regressor=regressor,
                    strategy=strategy,
                    fold=fold,
                    seed=seed,
                    selected_features=selected_features,
                    imputation_values=imputation_values,
                    threshold_pct=bin_thresh,
                )

                # --- EVALUATION 2: On the single, static final test set ---
                if test_df is not None:
                    X_test, y_bin_test, y_cont_test = self._prepare_strategy_data(test_df, target_col, bin_thresh)
                    if X_test is not None:
                        # Apply the imputation learned from the training set
                        X_test_sel = X_test[selected_features].fillna(imputation_values)
                        
                        if self.use_regressor:
                            test_metrics = evaluate_fold(
                                classifier, regressor, X_test_sel, y_bin_test, y_cont_test,
                                use_min_signal_gate=self.enable_min_signal_gate,
                                signal_prob_threshold=self.min_signal_gate_threshold
                            )
                        else:
                            test_metrics = evaluate_classifier_only(
                                classifier, X_test_sel, y_bin_test, y_cont_test,
                                use_min_signal_gate=self.enable_min_signal_gate,
                                signal_prob_threshold=self.min_signal_gate_threshold
                            )
                        if test_metrics:
                            all_test_results.append({'Timepoint': timepoint, 'TP': tp, 'SL': sl, 'Threshold': bin_thresh, 'Seed': seed, 'Fold': fold, **test_metrics})

        # Save results to two separate files
        # Append tuned hyperparameters as a suffix in the output filenames for traceability
        suffix = self._format_hparam_suffix(self.best_hyperparams)
        mode_suffix = "_clsOnly" if not self.use_regressor else "_clsReg"
        if all_validation_results:
            save_strategy_results(pd.DataFrame(all_validation_results), self.output_dir, f"{model_type}_Validation_Metrics{suffix}{mode_suffix}")
        if all_test_results:
            save_strategy_results(pd.DataFrame(all_test_results), self.output_dir, f"{model_type}_Test_Metrics{suffix}{mode_suffix}")

    def load_model(self, strategy, fold, seed, threshold_pct, model_type='both'):
            """Load a saved model for inference with a specific label threshold."""
            strategy_str = self._get_strategy_string(strategy)
            thr_str = str(threshold_pct).replace('.', 'p')
            model_dir = self.models_base_path / strategy_str / f"thr_{thr_str}" / f"fold_{fold}" / f"seed_{seed}"
            
            if not model_dir.exists():
                # Fallback to legacy path without threshold partition
                legacy_dir = self.models_base_path / strategy_str / f"fold_{fold}" / f"seed_{seed}"
                if legacy_dir.exists():
                    model_dir = legacy_dir
                else:
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