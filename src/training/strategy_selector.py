# file: src/training/strategy_selector.py
"""Strategy selection against the success criteria from the rebuild plan.

Reads a walk-forward *test-set* metrics workbook (as produced by
``save_strategy_results`` -> ``{model_type}_Test_Metrics_Summary.xlsx``),
aggregates per strategy across folds and seeds, applies the pass/fail
thresholds, and picks the best passing strategy by net Sharpe.

Success criteria (computed on the held-out test set, net of costs):
    Portfolio Sharpe (Net) >= 0.75, Portfolio Max Drawdown <= 0.15,
    Profit Factor >= 1.30, Mean Alpha (Net) > 0 (alpha vs SPX).

Sharpe and drawdown use the portfolio-realistic daily-rebalanced curve
(see ``portfolio_daily_returns``), not the per-trade sequential curve which
overstates both. Per-trade ``Sharpe (Net)`` / ``Max Drawdown`` and ``Win
Rate`` remain in the table as diagnostics.

The 50% win-rate gate was dropped: this is a low-hit-rate / high-payoff
TP-SL strategy where a few large winners drive Sharpe and profit factor, so
a majority-win threshold is the wrong shape.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns that identify a strategy; everything else numeric is a metric.
STRATEGY_KEYS = ["Timepoint", "TP", "SL", "Threshold"]

# (metric column, comparison, threshold, strict)
#   "min" -> value >= threshold ; "max" -> value <= threshold
#   strict=True makes it a strict inequality (> / <).
DEFAULT_THRESHOLDS = [
    ("Portfolio Sharpe (Net)", "min", 0.75, False),
    ("Portfolio Max Drawdown", "max", 0.15, False),
    ("Profit Factor", "min", 1.30, False),
    # Targets are alpha-vs-SPX by construction, so positive net alpha == alpha > 0.
    # Mean (not median) over executed trades: the cost haircut zeroes many
    # signals, pinning the median at 0.
    ("Mean Alpha (Net)", "min", 0.0, True),
]

SELECTION_METRIC = "Portfolio Sharpe (Net)"

# Thresholds for the DAILY-MTM ensemble backtest report (ensemble_backtest.py),
# whose columns are the lower-cased portfolio metrics (sharpe, max_drawdown,
# total_alpha) rather than the per-(strategy,seed,fold) test-metrics columns.
# Same (column, comparison, threshold, strict) shape as DEFAULT_THRESHOLDS.
BACKTEST_THRESHOLDS = [
    ("sharpe", "min", 0.75, False),
    ("max_drawdown", "max", 0.15, False),
    ("total_alpha", "min", 0.0, True),
]

BACKTEST_SELECTION_METRIC = "sharpe"


def _strategy_string(row) -> str:
    """Folder-safe strategy id, matching ModelTrainer._get_strategy_string."""
    tp = str(row["TP"]).replace(".", "p")
    sl = str(row["SL"]).replace(".", "p")
    return f"{row['Timepoint']}_tp{tp}_sl{sl}"


def load_metrics(metrics_xlsx, sheet_name: str = "Raw Results") -> pd.DataFrame:
    """Load the per-(strategy, seed, fold) metric rows from a results workbook."""
    return pd.read_excel(Path(metrics_xlsx), sheet_name=sheet_name)


def aggregate_by_strategy(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean each numeric metric across folds and seeds; one row per strategy."""
    if raw.empty:
        return raw.copy()
    metric_cols = [
        c
        for c in raw.columns
        if c not in STRATEGY_KEYS
        and c not in ("Seed", "Fold")
        and pd.api.types.is_numeric_dtype(raw[c])
    ]
    agg = raw.groupby(STRATEGY_KEYS, as_index=False)[metric_cols].mean()
    agg.insert(0, "Strategy", agg.apply(_strategy_string, axis=1))
    return agg


def _passes(value, comparison: str, threshold: float, strict: bool) -> bool:
    if pd.isna(value):
        return False
    if comparison == "min":
        return value > threshold if strict else value >= threshold
    if comparison == "max":
        return value < threshold if strict else value <= threshold
    raise ValueError(f"Unknown comparison: {comparison}")


def evaluate_strategies(metrics, thresholds=None) -> pd.DataFrame:
    """Score each strategy against the success criteria.

    ``metrics`` may be a path to a results workbook or an already-loaded
    raw-metrics DataFrame. Returns the per-strategy aggregate with one
    boolean ``Pass: <metric>`` column per criterion plus an overall ``Pass``,
    sorted by the selection metric (descending).
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if isinstance(metrics, (str, Path)):
        metrics = load_metrics(metrics)
    agg = aggregate_by_strategy(metrics)
    if agg.empty:
        return agg

    pass_cols = []
    for metric, comparison, threshold, strict in thresholds:
        col = f"Pass: {metric}"
        if metric not in agg.columns:
            agg[col] = False
        else:
            agg[col] = agg[metric].apply(
                lambda v, c=comparison, t=threshold, s=strict: _passes(v, c, t, s)
            )
        pass_cols.append(col)
    agg["Pass"] = agg[pass_cols].all(axis=1)

    sort_key = SELECTION_METRIC if SELECTION_METRIC in agg.columns else pass_cols[0]
    return agg.sort_values(sort_key, ascending=False).reset_index(drop=True)


def evaluate_backtest(backtest, thresholds=None) -> pd.DataFrame:
    """Score a daily-MTM ensemble backtest report against the success criteria.

    ``backtest`` is the per-strategy DataFrame from
    ``ensemble_backtest.run_all`` (or a path to its ``*_Backtest_Metrics.xlsx``).
    Unlike ``evaluate_strategies`` there is nothing to aggregate — each row is
    already one strategy. Gates on net-of-cost portfolio ``sharpe`` >= 0.75,
    ``max_drawdown`` <= 0.15, and ``total_alpha`` > 0 (strict). Adds one
    ``Pass: <metric>`` column per criterion plus an overall ``Pass``, sorted by
    ``sharpe`` (descending).
    """
    thresholds = thresholds or BACKTEST_THRESHOLDS
    if isinstance(backtest, (str, Path)):
        backtest = pd.read_excel(Path(backtest), sheet_name="Backtest Metrics")
    report = backtest.copy()
    if report.empty:
        return report

    pass_cols = []
    for metric, comparison, threshold, strict in thresholds:
        col = f"Pass: {metric}"
        if metric not in report.columns:
            report[col] = False
        else:
            report[col] = report[metric].apply(
                lambda v, c=comparison, t=threshold, s=strict: _passes(v, c, t, s)
            )
        pass_cols.append(col)
    report["Pass"] = report[pass_cols].all(axis=1)

    sort_key = (
        BACKTEST_SELECTION_METRIC
        if BACKTEST_SELECTION_METRIC in report.columns
        else pass_cols[0]
    )
    return report.sort_values(sort_key, ascending=False).reset_index(drop=True)


def select_best(report: pd.DataFrame, metric: str = SELECTION_METRIC):
    """Best passing strategy by ``metric`` (default net Sharpe), or None."""
    if report.empty or "Pass" not in report.columns or metric not in report.columns:
        return None
    passing = report[report["Pass"]]
    if passing.empty:
        return None
    return passing.loc[passing[metric].idxmax()].to_dict()


def compare_models(lgbm_report, tabpfn_report, metric: str = SELECTION_METRIC) -> dict:
    """Compare the best passing strategy from each model family.

    LightGBM is the incumbent; TabPFN is adopted only if it strictly beats
    LightGBM on ``metric`` (and itself passes the criteria).
    """
    lgbm_best = select_best(lgbm_report, metric)
    tabpfn_best = select_best(tabpfn_report, metric)

    if lgbm_best is None and tabpfn_best is None:
        winner = None
    elif tabpfn_best is None:
        winner = "LightGBM"
    elif lgbm_best is None:
        winner = "TabPFN"
    else:
        winner = "TabPFN" if tabpfn_best[metric] > lgbm_best[metric] else "LightGBM"

    return {
        "metric": metric,
        "winner": winner,
        "lgbm_best": lgbm_best,
        "tabpfn_best": tabpfn_best,
    }


def paper_to_live_gate(
    live_metrics: dict,
    thresholds=None,
    min_trades: int = 30,
    min_days: int = 90,
    win_rate_haircut: float = 0.15,
    slippage_mult: float = 2.0,
) -> dict:
    """Advisory paper->live promotion gate.

    ``live_metrics`` holds realized paper-track metrics, e.g. the success
    criteria plus ``Num Closed Trades``, ``Track Days``, ``Avg Cost (bps)``.
    Returns per-criterion pass flags, a sample-size check, a stress check
    (win rate -15pts and slippage x2 still profitable), and an overall
    recommendation. Advisory only — never flips PAPER_MODE.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    checks = {
        metric: _passes(live_metrics.get(metric), comparison, threshold, strict)
        for metric, comparison, threshold, strict in thresholds
    }

    n_trades = live_metrics.get("Num Closed Trades", 0) or 0
    track_days = live_metrics.get("Track Days", 0) or 0
    sample_ok = n_trades >= min_trades and track_days >= min_days

    win_rate = live_metrics.get("Win Rate")
    stressed_win = win_rate - win_rate_haircut if win_rate is not None else float("nan")
    base_cost_bps = live_metrics.get("Avg Cost (bps)", 0.0) or 0.0
    base_alpha = live_metrics.get("Mean Alpha (Net)")
    stressed_alpha = (
        base_alpha - (slippage_mult - 1.0) * base_cost_bps / 1e4
        if base_alpha is not None
        else float("nan")
    )
    stress_ok = (
        not pd.isna(stressed_win)
        and stressed_win > 0.0
        and not pd.isna(stressed_alpha)
        and stressed_alpha > 0.0
    )

    criteria_ok = all(checks.values())
    return {
        "criteria_pass": checks,
        "criteria_ok": criteria_ok,
        "sample_ok": sample_ok,
        "stress_ok": stress_ok,
        "recommend_promote": bool(criteria_ok and sample_ok and stress_ok),
        "note": "Advisory only - PAPER_MODE is never auto-flipped.",
    }


def write_report(report: pd.DataFrame, out_path, best=None) -> Path:
    """Write the selection report to an Excel workbook."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="Strategy Selection", index=False)
        if best is not None:
            pd.DataFrame([best]).to_excel(
                writer, sheet_name="Best Strategy", index=False
            )
    return out_path
