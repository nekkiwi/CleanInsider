# file: src/training/training_helpers.py

import os
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef
from lightgbm import LGBMClassifier

def calculate_spread_cost(X_fold: pd.DataFrame) -> pd.Series:
    """
    Estimates a per-trade transaction cost in decimal form (e.g., 0.001 for 10 bps)
    based on a composite score of liquidity and size proxy features.
    
    Args:
        X_fold (pd.DataFrame): The feature dataframe for the fold, containing the raw feature values.

    Returns:
        pd.Series: A series of estimated transaction costs, indexed the same as X_fold.
    """
    cost_proxies = pd.DataFrame(index=X_fold.index)
    
    # Invert size proxies so that smaller values (higher cost) get higher scores
    if 'Assets_q-1' in X_fold.columns:
        cost_proxies['size_inv'] = -1 * X_fold['Assets_q-1']
    
    # Value and profitability proxies
    if 'StockholdersEquity_q-1_per_MarketCap' in X_fold.columns:
        cost_proxies['value'] = X_fold['StockholdersEquity_q-1_per_MarketCap']
    if 'NetIncomeLoss_q-2_per_MarketCap' in X_fold.columns:
        cost_proxies['profitability'] = X_fold['NetIncomeLoss_q-2_per_MarketCap']
    if 'log_Value' in X_fold.columns:
        cost_proxies['trade_size'] = X_fold['log_Value']

    if cost_proxies.empty:
        # If no proxy features are found, return a default flat cost of 5 bps
        return pd.Series(0.0005, index=X_fold.index)

    # Normalize each proxy to be on a similar scale (0 to 1)
    for col in cost_proxies.columns:
        cost_proxies[col] = (cost_proxies[col] - cost_proxies[col].min()) / (cost_proxies[col].max() - cost_proxies[col].min())

    # Create a composite score (simple average) and fill any NaNs
    composite_score = cost_proxies.mean(axis=1).fillna(0.5)

    # Scale the score to a realistic bps range, e.g., 5 to 25 bps
    min_cost_bps = 5
    max_cost_bps = 25
    scaled_cost_bps = min_cost_bps + (composite_score * (max_cost_bps - min_cost_bps))
    
    # Convert bps to decimal form (e.g., 10 bps -> 0.0010)
    return scaled_cost_bps / 10000


def select_features_for_fold(X: pd.DataFrame, y: pd.Series, top_n: int, seed: int) -> list:
    """Selects the top N features based on LightGBM feature importance."""
    # This function remains unchanged
    if X.empty: return []
    X_imputed = X.fillna(X.median())
    feature_ranker = LGBMClassifier(n_estimators=100, random_state=seed, n_jobs=-1, verbosity=-1)
    feature_ranker.fit(X_imputed, y)
    importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_ranker.feature_importances_})
    top_features = importances_df.sort_values(by='Importance', ascending=False).head(top_n)
    return top_features['Feature'].tolist()

def annualize_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates the annualized Sharpe ratio."""
    # This function remains unchanged
    if returns.std() == 0 or len(returns) < 2: return np.nan
    excess_returns = returns - risk_free_rate / 252
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)

def adjusted_sharpe_ratio(sharpe: float, num_signals: int, target_signals: int = 100) -> float:
    """Adjusts the Sharpe Ratio for the number of trades."""
    # This function remains unchanged
    if pd.isna(sharpe) or num_signals <= 0: return 0.0
    return sharpe * min(1.0, np.sqrt(num_signals / target_signals))

def find_optimal_threshold(predicted_returns: pd.Series, actual_returns: pd.Series, costs: pd.Series) -> dict:
    """
    Finds the regressor output threshold that maximizes adjusted Sharpe on a validation set,
    now accounting for transaction costs.
    """
    best_score, best_threshold = -np.inf, np.nan
    for percentile in range(1, 100):
        threshold = np.percentile(predicted_returns, percentile)
        final_selection_idx = predicted_returns[predicted_returns >= threshold].index
        
        # Subtract costs from the returns of the selected trades
        final_returns_after_cost = actual_returns.loc[final_selection_idx] - costs.loc[final_selection_idx]
        
        if len(final_returns_after_cost) < 10: continue
        score = adjusted_sharpe_ratio(annualize_sharpe_ratio(final_returns_after_cost), len(final_returns_after_cost))
        if pd.notna(score) and score > best_score:
            best_score, best_threshold = score, threshold
    return {'validation_score': best_score, 'optimal_threshold': best_threshold}

def evaluate_test_fold(classifier, regressor, optimal_threshold, X_ts, y_bin_ts, y_cont_ts, costs_ts: pd.Series) -> dict or None:
    """
    Evaluates a single test fold, applying the estimated spread cost to the final returns.
    """
    if X_ts.empty or pd.isna(optimal_threshold): return None
    
    buy_signals = classifier.predict(X_ts)
    if buy_signals.sum() == 0: return None

    pos_class_idx = X_ts.index[buy_signals == 1]
    
    predicted_returns = pd.Series(regressor.predict(X_ts.loc[pos_class_idx]), index=pos_class_idx)
    final_selection_idx = predicted_returns[predicted_returns >= optimal_threshold].index
    
    # Get the gross returns before applying costs
    final_returns_gross = y_cont_ts.loc[final_selection_idx]
    if final_returns_gross.empty: return None

    # --- APPLY SPREAD COST ---
    final_returns_net = final_returns_gross - costs_ts.loc[final_selection_idx]
    
    # Calculate performance metrics on the NET returns
    sharpe_final_net = annualize_sharpe_ratio(final_returns_net)
    adj_sharpe_final_net = adjusted_sharpe_ratio(sharpe_final_net, len(final_returns_net))
    
    return {
        'Adj Sharpe (Net)': adj_sharpe_final_net,
        'Sharpe (Net)': sharpe_final_net,
        'Num Signals (Final)': len(final_returns_net),
        'Avg Cost (bps)': costs_ts.loc[final_selection_idx].mean() * 10000,
        'MCC (Classifier)': matthews_corrcoef(y_bin_ts, buy_signals)
    }

# save_strategy_results remains unchanged
def save_strategy_results(results_df: pd.DataFrame, stats_dir: str, model_name: str, category: str):
    # This function is correct as is.
    if results_df.empty:
        print("[INFO] No results to save.")
        return
    output_path = os.path.join(stats_dir, f"{model_name}_{category}_walk_forward_summary.xlsx")
    group_cols = ['Timepoint', 'TP', 'SL', 'Threshold']
    # If it's the validation results, we also group by Fold
    if 'Fold' in results_df.columns:
        group_cols.append('Fold')
    display_cols = [col for col in results_df.columns if col not in group_cols and col != 'Seed']
    mean_df = results_df.groupby(group_cols)[display_cols].mean().reset_index()
    std_df = results_df.groupby(group_cols)[display_cols].std().reset_index()
    mean_df.columns = [col if col in group_cols else f"{col} (Mean)" for col in mean_df.columns]
    std_df.columns = [col if col in group_cols else f"{col} (Std)" for col in std_df.columns]
    summary_df = pd.merge(mean_df, std_df, on=group_cols, how='left')
    summary_df.sort_values(by=['Timepoint', 'TP', 'SL', 'Fold'], inplace=True)
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Per-Fold Summary', index=False)
            results_df.to_excel(writer, sheet_name='Raw Seed Results', index=False)
        print(f"\n--- Strategy results saved to {output_path} ---")
    except Exception as e:
        print(f"[ERROR] Failed to save Excel file: {e}")
