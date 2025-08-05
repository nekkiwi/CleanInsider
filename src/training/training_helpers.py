# file: src/training/training_helpers.py

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef


def calculate_spread_cost(X_fold: pd.DataFrame) -> pd.Series:
    """
    Estimates a per-trade transaction cost based on a composite score of liquidity
    and size proxy features, using the definitive, correct feature names.
    """
    cost_proxies = pd.DataFrame(index=X_fold.index)

    # --- UPDATED TO USE YOUR FINAL FEATURE NAMES ---
    # Proxy for company size (larger assets -> lower cost)
    # We invert it so that smaller values (higher cost) get higher scores.
    if "FIN_Total Assets_Y1" in X_fold.columns:
        cost_proxies["size_inv"] = -1 * X_fold["FIN_Total Assets_Y1"]

    # Proxy for company value/stability
    if "FIN_Stockholders Equity_Y1" in X_fold.columns:
        cost_proxies["value"] = X_fold["FIN_Stockholders Equity_Y1"]

    # Proxy for profitability
    if "FIN_Net Income_Y1" in X_fold.columns:
        cost_proxies["profitability"] = X_fold["FIN_Net Income_Y1"]

    # Proxy for the size of the trade itself
    if "log_Value" in X_fold.columns:
        cost_proxies["trade_size"] = X_fold["log_Value"]
    # --- END OF UPDATES ---

    if cost_proxies.empty:
        # If no proxy features are found, return a default flat cost of 5 bps
        print("[WARN] No cost proxy features found. Using default 5bps cost.")
        return pd.Series(0.0005, index=X_fold.index)

    # Normalize each proxy to be on a similar scale (0 to 1)
    for col in cost_proxies.columns:
        cost_proxies[col] = (cost_proxies[col] - cost_proxies[col].min()) / (
            cost_proxies[col].max() - cost_proxies[col].min()
        )

    # Create a composite score (simple average) and fill any NaNs
    composite_score = cost_proxies.mean(axis=1).fillna(0.5)

    # Scale the score to a realistic bps range, e.g., 5 to 25 bps
    min_cost_bps, max_cost_bps = 5, 25
    scaled_cost_bps = min_cost_bps + (composite_score * (max_cost_bps - min_cost_bps))

    # Convert bps to decimal form (e.g., 10 bps -> 0.0010)
    return scaled_cost_bps / 10000


def select_features_for_fold(
    X: pd.DataFrame, y: pd.Series, top_n: int, seed: int
) -> list:
    """Selects the top N features based on LightGBM feature importance."""
    # This function is correct and remains unchanged.
    if X.empty:
        return []
    X_imputed = X.fillna(X.median())
    feature_ranker = LGBMClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1, verbosity=-1
    )
    feature_ranker.fit(X_imputed, y)
    importances_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": feature_ranker.feature_importances_}
    )
    return (
        importances_df.sort_values(by="Importance", ascending=False)
        .head(top_n)["Feature"]
        .tolist()
    )


def annualize_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates the annualized Sharpe ratio."""
    # This function is correct and remains unchanged.
    if returns.std() == 0 or len(returns) < 2:
        return np.nan
    excess_returns = returns - risk_free_rate / 252
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)


def adjusted_sharpe_ratio(
    sharpe: float, num_signals: int, target_signals: int = 100
) -> float:
    """Adjusts the Sharpe Ratio for the number of trades."""
    # This function is correct and remains unchanged.
    if pd.isna(sharpe) or num_signals <= 0:
        return 0.0
    return sharpe * min(1.0, np.sqrt(num_signals / target_signals))


def find_optimal_threshold(
    predicted_returns: pd.Series, actual_returns: pd.Series, costs: pd.Series
) -> dict:
    """Finds the regressor output threshold that maximizes adjusted Sharpe on a validation set."""
    best_score, best_threshold = -np.inf, np.nan
    for percentile in range(1, 100):
        threshold = np.percentile(predicted_returns, percentile)
        final_selection_idx = predicted_returns[predicted_returns >= threshold].index
        final_returns_after_cost = (
            actual_returns.loc[final_selection_idx] - costs.loc[final_selection_idx]
        )
        if len(final_returns_after_cost) < 10:
            continue
        score = adjusted_sharpe_ratio(
            annualize_sharpe_ratio(final_returns_after_cost),
            len(final_returns_after_cost),
        )
        if pd.notna(score) and score > best_score:
            best_score, best_threshold = score, threshold
    return {"validation_score": best_score, "optimal_threshold": best_threshold}


def evaluate_fold(
    classifier,
    regressor,
    optimal_threshold,
    X_eval,
    y_bin_eval,
    y_cont_eval,
    costs_eval,
):
    """Evaluates a model on a given dataset (validation or test) using a pre-determined optimal threshold."""
    if X_eval.empty or pd.isna(optimal_threshold):
        return None
    buy_signals = classifier.predict(X_eval)
    if buy_signals.sum() == 0:
        return None

    pos_class_idx = X_eval.index[buy_signals == 1]
    predicted_returns = pd.Series(
        regressor.predict(X_eval.loc[pos_class_idx]), index=pos_class_idx
    )

    final_selection_idx = predicted_returns[
        predicted_returns >= optimal_threshold
    ].index
    if final_selection_idx.empty:
        return None

    final_returns_net = (
        y_cont_eval.loc[final_selection_idx] - costs_eval.loc[final_selection_idx]
    )
    if final_returns_net.empty:
        return None

    sharpe_final_net = annualize_sharpe_ratio(final_returns_net)
    adj_sharpe_final_net = adjusted_sharpe_ratio(
        sharpe_final_net, len(final_returns_net)
    )

    return {
        "Adj Sharpe (Net)": adj_sharpe_final_net,
        "Sharpe (Net)": sharpe_final_net,
        "Num Signals (Final)": len(final_returns_net),
        "Avg Cost (bps)": costs_eval.loc[final_selection_idx].mean() * 10000,
        "MCC (Classifier)": matthews_corrcoef(y_bin_eval, buy_signals),
    }


def save_strategy_results(
    results_df: pd.DataFrame, stats_dir: Path, file_name_prefix: str
):
    """Saves strategy results to a distinctly named Excel file."""
    # This function is correct and remains unchanged.
    if results_df.empty:
        return
    group_cols = ["Timepoint", "TP", "SL", "Threshold"]
    if "Fold" in results_df.columns:
        group_cols.append("Fold")
    display_cols = [
        col for col in results_df.columns if col not in group_cols and col != "Seed"
    ]
    mean_df = results_df.groupby(group_cols)[display_cols].mean().reset_index()
    std_df = results_df.groupby(group_cols)[display_cols].std().reset_index()
    mean_df.columns = [
        col if col in group_cols else f"{col} (Mean)" for col in mean_df.columns
    ]
    std_df.columns = [
        col if col in group_cols else f"{col} (Std)" for col in std_df.columns
    ]
    summary_df = pd.merge(mean_df, std_df, on=group_cols, how="left")
    sort_cols = ["Timepoint", "TP", "SL"]
    if "Fold" in summary_df.columns:
        sort_cols.append("Fold")
    summary_df.sort_values(by=sort_cols, inplace=True)
    output_path = stats_dir / f"{file_name_prefix}_Summary.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        results_df.to_excel(writer, sheet_name="Raw Results", index=False)
    print(f"\n--- Strategy results saved to {output_path} ---")
