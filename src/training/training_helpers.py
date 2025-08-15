# file: src/training/training_helpers.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import matthews_corrcoef
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import hypergeom, spearmanr




def select_features_for_fold(X: pd.DataFrame, y: pd.Series, top_n: int, seed: int) -> list:
    """
    Selects the top N features based on LightGBM feature importance.
    Assumes X is already imputed.
    """
    if X.empty: return []
    feature_ranker = LGBMClassifier(n_estimators=100, random_state=seed, n_jobs=-1, verbosity=-1)
    feature_ranker.fit(X, y)
    importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_ranker.feature_importances_})
    return importances_df.sort_values(by='Importance', ascending=False).head(top_n)['Feature'].tolist()

def annualize_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates the annualized Sharpe ratio."""
    # This function is correct and remains unchanged.
    if returns.std() == 0 or len(returns) < 2: return np.nan
    excess_returns = returns - risk_free_rate / 252
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)

def adjusted_sharpe_ratio(sharpe: float, num_signals: int, target_signals: int = 100) -> float:
    """Adjusts the Sharpe Ratio for the number of trades."""
    # This function is correct and remains unchanged.
    if pd.isna(sharpe) or num_signals <= 0: return 0.0
    return sharpe * min(1.0, np.sqrt(num_signals / target_signals))

def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates the annualized Sortino ratio."""
    if returns.std() == 0 or len(returns) < 2:
        return np.nan
    rf_day = risk_free_rate / 252
    downside = returns[returns < rf_day] - rf_day
    if downside.empty:
        return np.nan
    downside_dev = np.sqrt(np.mean(np.square(downside)))
    if downside_dev == 0:
        return np.nan
    excess = returns.mean() - rf_day
    return (excess / downside_dev) * np.sqrt(252)

def max_drawdown(returns: pd.Series) -> float:
    """Computes max drawdown from a returns series."""
    if returns.empty:
        return np.nan
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())

def evaluate_classifier_only(classifier, X_eval, y_bin_eval, y_cont_eval,
                             use_min_signal_gate: bool = False,
                             signal_prob_threshold: float = 0.5):
    """
    Evaluate classifier-only metrics without a regressor.
    Returns Sharpe and Adj Sharpe for classifier-selected trades, and MCC.
    """
    if X_eval.empty:
        return None
    if use_min_signal_gate and hasattr(classifier, 'predict_proba'):
        try:
            proba = classifier.predict_proba(X_eval)[:, 1]
            buy_signals = (proba >= signal_prob_threshold).astype(int)
        except Exception:
            buy_signals = classifier.predict(X_eval)
    else:
        buy_signals = classifier.predict(X_eval)
    if buy_signals.sum() == 0:
        return None
    sel_idx = X_eval.index[buy_signals == 1]
    classifier_returns = y_cont_eval.loc[sel_idx]
    sharpe_classifier = annualize_sharpe_ratio(classifier_returns)
    adj_sharpe_classifier = adjusted_sharpe_ratio(sharpe_classifier, len(classifier_returns))
    sortino_cls = sortino_ratio(classifier_returns)
    mdd_cls = max_drawdown(classifier_returns)

    gt_hits_idx = X_eval.index[y_bin_eval == 1]
    pval_classifier = hypergeometric_pvalue(gt_hits_idx, sel_idx, len(X_eval))

    if not gt_hits_idx.empty:
        gt_returns = y_cont_eval.loc[gt_hits_idx]
        sharpe_gt = annualize_sharpe_ratio(gt_returns)
        adj_sharpe_gt = adjusted_sharpe_ratio(sharpe_gt, len(gt_returns))
        sortino_gt = sortino_ratio(gt_returns)
        mdd_gt = max_drawdown(gt_returns)
        num_signals_gt = int(len(gt_returns))
    else:
        sharpe_gt = np.nan
        adj_sharpe_gt = np.nan
        sortino_gt = np.nan
        mdd_gt = np.nan
        num_signals_gt = 0

    return {
        'Sharpe (Classifier)': sharpe_classifier,
        'Adj Sharpe (Classifier)': adj_sharpe_classifier,
        'Sortino (Classifier)': sortino_cls,
        'Max Drawdown (Classifier)': mdd_cls,
        'MCC (Classifier)': matthews_corrcoef(y_bin_eval, buy_signals),
        'GT-vs-Classifier p-value': pval_classifier,
        'Num Signals (Classifier)': int(len(classifier_returns)),
        'Sharpe (Ground Truth)': sharpe_gt,
        'Adj Sharpe (Ground Truth)': adj_sharpe_gt,
        'Sortino (Ground Truth)': sortino_gt,
        'Max Drawdown (Ground Truth)': mdd_gt,
        'Num Signals (Ground Truth)': num_signals_gt,
    }

def calculate_position_sizes(predicted_returns: pd.Series, min_size: float = 0.25, max_size: float = 1.0) -> pd.Series:
    """
    Scales regressor outputs to a position size between min_size and max_size.
    """
    if predicted_returns.empty:
        return pd.Series(dtype=float)
    
    # Min-Max Scaling
    min_pred = predicted_returns.min()
    max_pred = predicted_returns.max()
    
    if max_pred == min_pred:
        # If all predictions are the same, assign the average size
        return pd.Series( (min_size + max_size) / 2, index=predicted_returns.index)
        
    scaled_preds = (predicted_returns - min_pred) / (max_pred - min_pred)
    
    # Scale to the desired range [min_size, max_size]
    position_sizes = min_size + scaled_preds * (max_size - min_size)
    
    return position_sizes

def hypergeometric_pvalue(gt_hits_idx, selected_idx, population_size):
    """
    Compute the p-value for the overlap between ground truth hits and selected signals using the hypergeometric test.
    - gt_hits_idx: indices of ground truth hits (set or pd.Index)
    - selected_idx: indices of selected signals (set or pd.Index)
    - population_size: total number of samples
    """
    gt_hits = set(gt_hits_idx)
    selected = set(selected_idx)
    n_gt_hits = len(gt_hits)
    n_selected = len(selected)
    n_overlap = len(gt_hits & selected)
    if n_gt_hits == 0 or n_selected == 0 or population_size == 0:
        return np.nan
    # P-value: probability of getting at least n_overlap hits by chance
    rv = hypergeom(population_size, n_gt_hits, n_selected)
    pval = rv.sf(n_overlap - 1)  # sf is 1-cdf, so this is P(X >= n_overlap)
    return pval

def evaluate_fold(classifier, regressor, X_eval, y_bin_eval, y_cont_eval, use_min_signal_gate: bool = False, signal_prob_threshold: float = 0.5):
    """
    Evaluates a model using fractional sizing based on the regressor's output.
    NOTE: The 'optimal_threshold' parameter has been removed.
    """
    if X_eval.empty or regressor is None: return None
    
    # STAGE 1: Get all buy signals from the classifier (the "gatekeeper")
    if use_min_signal_gate and hasattr(classifier, 'predict_proba'):
        try:
            proba = classifier.predict_proba(X_eval)[:, 1]
            buy_signals = (proba >= signal_prob_threshold).astype(int)
        except Exception:
            buy_signals = classifier.predict(X_eval)
    else:
        buy_signals = classifier.predict(X_eval)
    if buy_signals.sum() == 0: return None
    
    pos_class_idx = X_eval.index[buy_signals == 1]
    
    # STAGE 2: Predict returns for the classifier's selections
    predicted_returns = pd.Series(regressor.predict(X_eval.loc[pos_class_idx]), index=pos_class_idx)
    if predicted_returns.empty: return None

    # NEW: Calculate position sizes based on regressor's predicted returns
    position_sizes = calculate_position_sizes(predicted_returns)

    # --- FINAL PORTFOLIO CALCULATION (WEIGHTED, NO EXTRA COSTS) ---
    # Note: We now use pos_class_idx, the full set of classifier signals
    actual_returns = y_cont_eval.loc[pos_class_idx]
    final_returns_net = (actual_returns * position_sizes)
    
    if final_returns_net.empty: return None

    # --- METRICS ---
    # Standard metrics are now calculated on the weighted portfolio returns
    sharpe_final_net = annualize_sharpe_ratio(final_returns_net)
    adj_sharpe_final_net = adjusted_sharpe_ratio(sharpe_final_net, len(final_returns_net))
    sortino_final_net = sortino_ratio(final_returns_net)
    mdd_final_net = max_drawdown(final_returns_net)

    # Calculate Information Coefficient (IC)
    # Use actual_returns BEFORE costs to measure pure prediction skill
    ic, _ = spearmanr(position_sizes, y_cont_eval.loc[pos_class_idx])
    
    # Calculate Capital Utilization
    avg_position_size = position_sizes.mean()

    # Calculate Profit Concentration
    # Sort trades by their net profit
    sorted_net_returns = final_returns_net.sort_values(ascending=False)
    top_10_percent_count = int(len(sorted_net_returns) * 0.10)
    profit_from_top_10_pct = sorted_net_returns.head(top_10_percent_count).sum()
    total_profit = sorted_net_returns.sum()
    profit_concentration = profit_from_top_10_pct / total_profit if total_profit > 0 else 0

    # Classifier-only metrics (on all buy_signals)
    if buy_signals.sum() > 1:
        # Only compute if there are at least 2 signals
        classifier_returns = y_cont_eval.loc[X_eval.index[buy_signals == 1]]
        sharpe_classifier = annualize_sharpe_ratio(classifier_returns)
        adj_sharpe_classifier = adjusted_sharpe_ratio(sharpe_classifier, len(classifier_returns))
        sortino_classifier = sortino_ratio(classifier_returns)
        mdd_classifier = max_drawdown(classifier_returns)
    else:
        sharpe_classifier = np.nan
        adj_sharpe_classifier = np.nan
        sortino_classifier = np.nan
        mdd_classifier = np.nan

    # Hypergeometric p-value for ground truth hits vs classifier hits
    gt_hits_idx = X_eval.index[y_bin_eval == 1]
    classifier_hits_idx = X_eval.index[buy_signals == 1]
    pval_classifier = hypergeometric_pvalue(gt_hits_idx, classifier_hits_idx, len(X_eval))

    # --- MEDIAN RETURN METRICS ---
    # Median return for all potential trades identified by the binary target
    median_alpha_gt = y_cont_eval.loc[gt_hits_idx].median() if not gt_hits_idx.empty else np.nan
    # Ground truth metrics
    if not gt_hits_idx.empty:
        gt_returns = y_cont_eval.loc[gt_hits_idx]
        sharpe_gt = annualize_sharpe_ratio(gt_returns)
        adj_sharpe_gt = adjusted_sharpe_ratio(sharpe_gt, len(gt_returns))
        sortino_gt = sortino_ratio(gt_returns)
        mdd_gt = max_drawdown(gt_returns)
        num_signals_gt = int(len(gt_returns))
    else:
        sharpe_gt = np.nan
        adj_sharpe_gt = np.nan
        sortino_gt = np.nan
        mdd_gt = np.nan
        num_signals_gt = 0
    
    # Median return for all trades selected by the classifier
    median_alpha_classifier = y_cont_eval.loc[classifier_hits_idx].median() if not classifier_hits_idx.empty else np.nan
    
    # Median of the final, weighted net returns
    median_alpha_final = final_returns_net.median() if not final_returns_net.empty else np.nan

    return {
        'Adj Sharpe (Net)': adj_sharpe_final_net,
        'Sharpe (Net)': sharpe_final_net,
        'Sortino (Net)': sortino_final_net,
        'Max Drawdown (Net)': mdd_final_net,
        'Num Signals (Final)': len(final_returns_net),
        # No explicit trading cost modeling; targets are already spread-adjusted
        'MCC (Classifier)': matthews_corrcoef(y_bin_eval, buy_signals),
        'Sharpe (Classifier)': sharpe_classifier,
        'Adj Sharpe (Classifier)': adj_sharpe_classifier,
        'Sortino (Classifier)': sortino_classifier,
        'Max Drawdown (Classifier)': mdd_classifier,
        'GT-vs-Classifier p-value': pval_classifier,
        'Information Coefficient': ic,
        'Avg Position Size': avg_position_size,
        'Profit Concentration (Top 10%)': profit_concentration,
        'Median Alpha (Ground Truth)': median_alpha_gt,
        'Median Alpha (Classifier)': median_alpha_classifier,
        'Median Alpha (Final Net)': median_alpha_final,
        'Sharpe (Ground Truth)': sharpe_gt,
        'Adj Sharpe (Ground Truth)': adj_sharpe_gt,
        'Sortino (Ground Truth)': sortino_gt,
        'Max Drawdown (Ground Truth)': mdd_gt,
        'Num Signals (Ground Truth)': num_signals_gt,
    }

def save_strategy_results(results_df: pd.DataFrame, stats_dir: Path, file_name_prefix: str):
    """Saves strategy results to a distinctly named Excel file."""
    # This function is correct and remains unchanged.
    if results_df.empty: return
    group_cols = ['Timepoint', 'TP', 'SL', 'Threshold']
    if 'Fold' in results_df.columns:
        group_cols.append('Fold')
    display_cols = [col for col in results_df.columns if col not in group_cols and col != 'Seed']
    mean_df = results_df.groupby(group_cols)[display_cols].mean().reset_index()
    std_df = results_df.groupby(group_cols)[display_cols].std().reset_index()
    mean_df.columns = [col if col in group_cols else f"{col} (Mean)" for col in mean_df.columns]
    std_df.columns = [col if col in group_cols else f"{col} (Std)" for col in std_df.columns]
    summary_df = pd.merge(mean_df, std_df, on=group_cols, how='left')
    sort_cols = ['Timepoint', 'TP', 'SL']
    if 'Fold' in summary_df.columns:
        sort_cols.append('Fold')
    summary_df.sort_values(by=sort_cols, inplace=True)
    output_path = stats_dir / f"{file_name_prefix}_Summary.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        results_df.to_excel(writer, sheet_name='Raw Results', index=False)
    print(f"\n--- Strategy results saved to {output_path} ---")

