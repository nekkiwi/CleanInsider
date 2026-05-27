# file: src/training/training_helpers.py

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import hypergeom, spearmanr
from sklearn.metrics import matthews_corrcoef


def select_features_for_fold(
    X: pd.DataFrame, y: pd.Series, top_n: int, seed: int
) -> list:
    """
    Selects the top N features based on LightGBM feature importance.
    Assumes X is already imputed.
    """
    if X.empty:
        return []
    feature_ranker = LGBMClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1, verbosity=-1
    )
    feature_ranker.fit(X, y)
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


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates the Sortino ratio (uses downside deviation)."""
    if len(returns) < 2:
        return np.nan
    excess = returns - risk_free_rate / 252
    downside = excess[excess < 0]
    if downside.std() == 0:
        return np.nan
    return excess.mean() / downside.std() * np.sqrt(252)


def adjusted_sharpe_ratio(
    sharpe: float, num_signals: int, target_signals: int = 100
) -> float:
    """Adjusts the Sharpe Ratio for the number of trades."""
    # This function is correct and remains unchanged.
    if pd.isna(sharpe) or num_signals <= 0:
        return 0.0
    return sharpe * min(1.0, np.sqrt(num_signals / target_signals))


def calculate_position_sizes(
    predicted_returns: pd.Series, min_size: float = 0.25, max_size: float = 1.0
) -> pd.Series:
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
        return pd.Series((min_size + max_size) / 2, index=predicted_returns.index)

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


def horizon_business_days(timepoint: str) -> int:
    """Map a strategy timepoint string (e.g. '1w', '2w', '1m') to business days.

    1 week -> 5, 1 month -> 21 trading days. Returns 0 for unknown input.
    """
    if not timepoint:
        return 0
    tp = str(timepoint).strip().lower()
    unit = tp[-1]
    try:
        n = int(tp[:-1])
    except ValueError:
        n = 1
    return {"w": n * 5, "m": n * 21, "d": n}.get(unit, 0)


def portfolio_daily_returns(entry_dates, total_returns, weights, horizon_days):
    """Conviction-weighted, daily-rebalanced portfolio return series.

    Each trade's total net return is spread geometrically across
    ``horizon_days`` business days from its entry (filing) date; on each day
    the active trades are combined as a weight-normalized (fully-invested)
    basket. Days with no active position are flat (cash). This turns the set of
    overlapping trades into a genuine daily series, so Sharpe annualizes
    correctly and drawdown reflects diversification rather than a single-bet
    sequential curve.

    Returns a numpy array of daily returns (empty if inputs are unusable).
    """
    entry_dates = pd.to_datetime(pd.Series(entry_dates)).dropna()
    if entry_dates.empty or not horizon_days or horizon_days <= 0:
        return np.array([])
    total_returns = pd.Series(total_returns).reindex(entry_dates.index)
    weights = pd.Series(weights).reindex(entry_dates.index).to_numpy(dtype=float)
    daily_per_trade = (1.0 + total_returns.to_numpy()) ** (1.0 / horizon_days) - 1.0

    bdays = pd.bdate_range(
        entry_dates.min(),
        entry_dates.max() + pd.tseries.offsets.BDay(horizon_days),
    )
    if len(bdays) == 0:
        return np.array([])
    start_pos = bdays.searchsorted(entry_dates.to_numpy())

    num = np.zeros(len(bdays))
    den = np.zeros(len(bdays))
    contrib = weights * daily_per_trade
    for offset in range(horizon_days):
        idx = start_pos + offset
        valid = idx < len(bdays)
        np.add.at(num, idx[valid], contrib[valid])
        np.add.at(den, idx[valid], weights[valid])
    # Divide only where a position is active; flat (0.0) on cash days. Avoids
    # the 0/0 RuntimeWarning that np.where(num/den) would trigger on cash days.
    daily = np.zeros(len(bdays))
    active = den > 0
    daily[active] = num[active] / den[active]
    return daily


def evaluate_fold(
    classifier,
    regressor,
    X_eval,
    y_bin_eval,
    y_cont_eval,
    costs_eval,
    dates_eval=None,
    horizon_days=None,
):
    """
    Evaluates a model using fractional sizing based on the regressor's output.
    NOTE: The 'optimal_threshold' parameter has been removed.

    When ``dates_eval`` (entry/filing dates aligned to ``X_eval.index``) and
    ``horizon_days`` are provided, portfolio-realistic ``Portfolio Sharpe
    (Net)`` / ``Portfolio Max Drawdown`` are also computed on a daily-rebalanced
    basket; otherwise those are NaN and only the per-trade diagnostics are
    returned.
    """
    if X_eval.empty or regressor is None:
        return None

    # STAGE 1: Get all buy signals from the classifier (the "gatekeeper")
    buy_signals = classifier.predict(X_eval)
    if buy_signals.sum() == 0:
        return None

    pos_class_idx = X_eval.index[buy_signals == 1]

    # STAGE 2: Predict returns for the classifier's selections
    predicted_returns = pd.Series(
        regressor.predict(X_eval.loc[pos_class_idx]), index=pos_class_idx
    )
    if predicted_returns.empty:
        return None

    # NEW: Calculate position sizes based on regressor's predicted returns
    position_sizes = calculate_position_sizes(predicted_returns)

    # --- FINAL PORTFOLIO CALCULATION (WEIGHTED) ---
    # Note: We now use pos_class_idx, the full set of classifier signals
    actual_returns = y_cont_eval.loc[pos_class_idx]
    trade_costs = costs_eval.loc[pos_class_idx]

    # One-way cost assumption: pay half the quoted spread per entry
    half_spread_costs = trade_costs * 0.5

    # Size haircut: scale position by 0.5% / spread, cap at 1
    size_haircut = (0.005 / half_spread_costs).clip(upper=1)
    effective_sizes = position_sizes * size_haircut

    # Zero-weight trades whose one-way cost exceeds 100 bps
    high_cost_mask = half_spread_costs > 0.01  # 100 bps
    effective_sizes.loc[high_cost_mask] = 0.0

    final_returns_net = (actual_returns * effective_sizes) - (
        half_spread_costs * effective_sizes
    )

    # Average cost actually paid (bps) weighted by position size
    if effective_sizes.sum() > 0:
        avg_cost_bps_paid = (
            (half_spread_costs * effective_sizes).sum() / effective_sizes.sum() * 1e4
        )
    else:
        avg_cost_bps_paid = np.nan

    if final_returns_net.empty:
        return None

    # --- METRICS ---
    # Standard metrics are now calculated on the weighted portfolio returns
    sharpe_final_net = annualize_sharpe_ratio(final_returns_net)
    sortino_final_net = calculate_sortino_ratio(final_returns_net)
    adj_sharpe_final_net = adjusted_sharpe_ratio(
        sharpe_final_net, len(final_returns_net)
    )

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
    profit_concentration = (
        profit_from_top_10_pct / total_profit if total_profit > 0 else 0
    )

    # Classifier-only metrics (on all buy_signals)
    if buy_signals.sum() > 1:
        # Only compute if there are at least 2 signals
        classifier_returns = (
            y_cont_eval.loc[X_eval.index[buy_signals == 1]]
            - costs_eval.loc[X_eval.index[buy_signals == 1]]
        )
        sharpe_classifier = annualize_sharpe_ratio(classifier_returns)
        sortino_classifier = calculate_sortino_ratio(classifier_returns)
        adj_sharpe_classifier = adjusted_sharpe_ratio(
            sharpe_classifier, len(classifier_returns)
        )
    else:
        sharpe_classifier = np.nan
        adj_sharpe_classifier = np.nan
        sortino_classifier = np.nan

    # Hypergeometric p-value for ground truth hits vs classifier hits
    gt_hits_idx = X_eval.index[y_bin_eval == 1]
    classifier_hits_idx = X_eval.index[buy_signals == 1]
    pval_classifier = hypergeometric_pvalue(
        gt_hits_idx, classifier_hits_idx, len(X_eval)
    )

    # --- MEDIAN RETURN METRICS ---
    # Median return for all potential trades identified by the binary target
    median_alpha_gt = (
        y_cont_eval.loc[gt_hits_idx].median() if not gt_hits_idx.empty else np.nan
    )

    # Median return for all trades selected by the classifier
    median_alpha_classifier = (
        y_cont_eval.loc[classifier_hits_idx].median()
        if not classifier_hits_idx.empty
        else np.nan
    )

    # Median of the final, weighted net returns
    median_alpha_final = (
        final_returns_net.median() if not final_returns_net.empty else np.nan
    )

    # --- TRADE-QUALITY / RISK METRICS (success-criteria inputs) ---
    # Measure over EXECUTED trades only. The cost haircut zero-weights some
    # signals (effective_size == 0); those carry no P&L, so counting them would
    # deflate win rate and drag mean alpha toward zero. "Win Rate" (over all
    # signals) is kept for reference; selection uses the executed variants.
    n_trades_final = len(final_returns_net)
    executed_mask = effective_sizes > 0
    executed_returns = final_returns_net[executed_mask]
    n_executed = len(executed_returns)

    win_rate = (final_returns_net > 0).mean() if n_trades_final > 0 else np.nan
    win_rate_executed = (executed_returns > 0).mean() if n_executed > 0 else np.nan
    mean_alpha_net = executed_returns.mean() if n_executed > 0 else np.nan

    gross_profit = executed_returns[executed_returns > 0].sum()
    gross_loss = -executed_returns[executed_returns < 0].sum()
    if gross_loss > 0:
        profit_factor = float(gross_profit / gross_loss)
    elif gross_profit > 0:
        profit_factor = 100.0  # no losing trades; cap to keep value finite
    else:
        profit_factor = np.nan
    if not pd.isna(profit_factor):
        profit_factor = float(np.clip(profit_factor, 0.0, 100.0))

    # Per-trade (sequential, single-bet) drawdown — kept as a diagnostic only.
    if n_executed > 0:
        equity_curve = (1.0 + executed_returns).cumprod()
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1.0
        max_drawdown = float(-drawdown.min())
    else:
        max_drawdown = np.nan

    # Portfolio-realistic Sharpe & MaxDD on a daily-rebalanced basket. Requires
    # entry dates + holding horizon; otherwise left NaN (e.g. in unit tests).
    portfolio_sharpe = np.nan
    portfolio_max_dd = np.nan
    portfolio_days = 0
    if dates_eval is not None and horizon_days and n_executed > 0:
        exec_idx = executed_returns.index
        daily = portfolio_daily_returns(
            entry_dates=pd.Series(dates_eval).reindex(exec_idx),
            total_returns=executed_returns,
            weights=effective_sizes.reindex(exec_idx),
            horizon_days=horizon_days,
        )
        if daily.size > 1 and np.std(daily) > 0:
            portfolio_sharpe = float(np.mean(daily) / np.std(daily) * np.sqrt(252))
            equity = np.cumprod(1.0 + daily)
            running_max = np.maximum.accumulate(equity)
            dd = equity / running_max - 1.0
            portfolio_max_dd = float(-dd.min())
            portfolio_days = int((daily != 0).sum())

    return {
        "Adj Sharpe (Net)": adj_sharpe_final_net,
        "Sharpe (Net)": sharpe_final_net,
        "Sortino (Net)": sortino_final_net,
        "Portfolio Sharpe (Net)": portfolio_sharpe,
        "Portfolio Max Drawdown": portfolio_max_dd,
        "Portfolio Days": portfolio_days,
        "Win Rate": win_rate,
        "Win Rate (Executed)": win_rate_executed,
        "Mean Alpha (Net)": mean_alpha_net,
        "Profit Factor": profit_factor,
        "Max Drawdown": max_drawdown,
        "Num Signals (Final)": len(final_returns_net),
        "Num Signals (Executed)": n_executed,
        "Avg Cost (bps)": avg_cost_bps_paid,
        "MCC (Classifier)": matthews_corrcoef(y_bin_eval, buy_signals),
        "Sharpe (Classifier)": sharpe_classifier,
        "Sortino (Classifier)": sortino_classifier,
        "Adj Sharpe (Classifier)": adj_sharpe_classifier,
        "GT-vs-Classifier p-value": pval_classifier,
        "Information Coefficient": ic,
        "Avg Position Size": avg_position_size,
        "Profit Concentration (Top 10%)": profit_concentration,
        "Median Alpha (Ground Truth)": median_alpha_gt,
        "Median Alpha (Classifier)": median_alpha_classifier,
        "Median Alpha (Final Net)": median_alpha_final,
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
