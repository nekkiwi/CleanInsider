# tests/test_strategy_selector.py
"""Unit tests for the Stage-2 success-criteria metrics and strategy selector.

Covers:
1. The new trade-quality metrics added to ``evaluate_fold`` (Win Rate,
   Profit Factor, Max Drawdown).
2. ``strategy_selector`` pass/fail evaluation, best-pick, model comparison,
   and the advisory paper->live gate -- all on synthetic metrics so no
   trained models are needed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path (repo convention; there is no conftest.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.strategy_selector import (
    aggregate_by_strategy,
    compare_models,
    evaluate_backtest,
    evaluate_strategies,
    paper_to_live_gate,
    select_best,
)
from src.training.training_helpers import evaluate_fold


# --------------------------------------------------------------------------- #
# evaluate_fold: new trade-quality metrics
# --------------------------------------------------------------------------- #
class _Stub:
    """Minimal model stub returning a fixed prediction vector."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def predict(self, X):
        return self._values[: len(X)]


def _make_eval_inputs(actual_returns, predicted_returns, cost=0.0002):
    idx = pd.RangeIndex(len(actual_returns))
    X_eval = pd.DataFrame({"f0": np.arange(len(idx), dtype=float)}, index=idx)
    classifier = _Stub(np.ones(len(idx)))  # every row is a buy signal
    regressor = _Stub(predicted_returns)
    y_bin_eval = pd.Series((np.asarray(actual_returns) > 0).astype(int), index=idx)
    y_cont_eval = pd.Series(actual_returns, index=idx, dtype=float)
    costs_eval = pd.Series(cost, index=idx, dtype=float)
    return classifier, regressor, X_eval, y_bin_eval, y_cont_eval, costs_eval


def test_evaluate_fold_mixed_winrate_and_profit_factor():
    # 3 winners (+10%), 2 losers (-5%); equal predicted -> equal sizing.
    actual = [0.10, 0.10, 0.10, -0.05, -0.05]
    predicted = [0.05] * 5
    metrics = evaluate_fold(*_make_eval_inputs(actual, predicted))

    assert metrics is not None
    expected = {
        "Win Rate",
        "Win Rate (Executed)",
        "Mean Alpha (Net)",
        "Profit Factor",
        "Max Drawdown",
    }
    assert expected <= set(metrics)
    assert metrics["Win Rate"] == 0.6
    assert metrics["Win Rate (Executed)"] == 0.6  # all signals executed here
    assert metrics["Mean Alpha (Net)"] > 0.0  # winners outweigh losers
    assert metrics["Profit Factor"] > 1.0
    assert np.isfinite(metrics["Profit Factor"])
    assert metrics["Max Drawdown"] >= 0.0
    # No dates passed -> portfolio metrics stay NaN.
    assert np.isnan(metrics["Portfolio Sharpe (Net)"])
    assert np.isnan(metrics["Portfolio Max Drawdown"])


def test_evaluate_fold_portfolio_metrics_with_dates():
    actual = [0.10, -0.05, 0.08, -0.04, 0.06, -0.03, 0.07, -0.02]
    predicted = [0.05, 0.04, 0.06, 0.03, 0.05, 0.02, 0.06, 0.03]
    classifier, regressor, X_eval, y_bin, y_cont, costs = _make_eval_inputs(
        actual, predicted
    )
    dates = pd.Series(
        pd.bdate_range("2023-01-02", periods=len(actual)), index=X_eval.index
    )
    metrics = evaluate_fold(
        classifier,
        regressor,
        X_eval,
        y_bin,
        y_cont,
        costs,
        dates_eval=dates,
        horizon_days=5,
    )
    assert metrics is not None
    assert np.isfinite(metrics["Portfolio Sharpe (Net)"])
    assert metrics["Portfolio Max Drawdown"] >= 0.0
    assert metrics["Portfolio Days"] > 0


def test_evaluate_fold_all_winners_caps_profit_factor_zero_drawdown():
    actual = [0.10, 0.08, 0.12, 0.05]
    predicted = [0.05] * 4
    metrics = evaluate_fold(*_make_eval_inputs(actual, predicted))

    assert metrics is not None
    assert metrics["Win Rate"] == 1.0
    # No losing trades -> profit factor capped at 100.0, equity monotonic -> no DD.
    assert metrics["Profit Factor"] == 100.0
    assert metrics["Max Drawdown"] == 0.0


# --------------------------------------------------------------------------- #
# strategy_selector: pass/fail + selection
# --------------------------------------------------------------------------- #
def _raw_row(timepoint, tp, sl, seed, fold, sharpe, win, pf, dd, alpha):
    # `sharpe`/`dd` populate the portfolio columns the selector gates on;
    # the per-trade columns mirror them for realism (diagnostics only).
    return {
        "Timepoint": timepoint,
        "TP": tp,
        "SL": sl,
        "Threshold": 2,
        "Seed": seed,
        "Fold": fold,
        "Portfolio Sharpe (Net)": sharpe,
        "Portfolio Max Drawdown": dd,
        "Sharpe (Net)": sharpe,
        "Max Drawdown": dd,
        "Win Rate": win,  # reported, no longer a gate
        "Profit Factor": pf,
        "Mean Alpha (Net)": alpha,
    }


def _synthetic_raw_metrics():
    rows = []
    # Strategy A: passes (mean Sharpe 1.0).
    rows.append(_raw_row("1w", 0.05, -0.05, 42, 1, 0.9, 0.55, 1.4, 0.10, 0.02))
    rows.append(_raw_row("1w", 0.05, -0.05, 42, 2, 1.1, 0.60, 1.6, 0.12, 0.03))
    # Strategy C: passes with higher Sharpe (mean 1.5) -> should be the best.
    rows.append(_raw_row("1m", 0.10, -0.10, 42, 1, 1.4, 0.58, 1.8, 0.09, 0.04))
    rows.append(_raw_row("1m", 0.10, -0.10, 42, 2, 1.6, 0.62, 2.0, 0.11, 0.05))
    # Strategy B: fails Sharpe (mean 0.30).
    rows.append(_raw_row("2w", 0.05, -0.05, 42, 1, 0.2, 0.52, 1.4, 0.10, 0.01))
    rows.append(_raw_row("2w", 0.05, -0.05, 42, 2, 0.4, 0.53, 1.5, 0.12, 0.01))
    return pd.DataFrame(rows)


def test_aggregate_collapses_folds_and_seeds():
    agg = aggregate_by_strategy(_synthetic_raw_metrics())
    assert len(agg) == 3  # three distinct strategies
    a = agg[agg["Strategy"] == "1w_tp0p05_sl-0p05"].iloc[0]
    assert a["Sharpe (Net)"] == 1.0  # mean of 0.9 and 1.1


def test_evaluate_and_select_best_picks_highest_passing_sharpe():
    report = evaluate_strategies(_synthetic_raw_metrics())
    assert int(report["Pass"].sum()) == 2  # A and C pass, B fails

    best = select_best(report)
    assert best is not None
    assert best["Strategy"] == "1m_tp0p1_sl-0p1"  # highest Sharpe among passing
    assert best["Pass"] is True or best["Pass"] == True  # noqa: E712


def test_failing_alpha_blocks_pass():
    # Strong everything but non-positive alpha must fail the strict alpha check.
    df = pd.DataFrame([_raw_row("1w", 0.05, -0.05, 42, 1, 2.0, 0.7, 3.0, 0.05, 0.0)])
    report = evaluate_strategies(df)
    assert report["Pass"].iloc[0] == False  # noqa: E712
    assert select_best(report) is None


def test_select_best_none_when_nothing_passes():
    df = pd.DataFrame([_raw_row("2w", 0.05, -0.05, 42, 1, 0.1, 0.4, 0.9, 0.30, -0.01)])
    report = evaluate_strategies(df)
    assert int(report["Pass"].sum()) == 0
    assert select_best(report) is None


# --------------------------------------------------------------------------- #
# evaluate_backtest (daily-MTM ensemble backtest report)
# --------------------------------------------------------------------------- #
def _backtest_report():
    """Two strategies: one passes; one fails purely on MaxDD."""
    return pd.DataFrame(
        [
            {
                "strategy_str": "1w_tp0p05_sl-0p05",
                "sharpe": 1.2,
                "max_drawdown": 0.10,  # within 0.15
                "total_alpha": 0.08,
            },
            {
                "strategy_str": "2w_tp0p05_sl-0p05",
                "sharpe": 1.5,  # higher Sharpe but...
                "max_drawdown": 0.25,  # ...fails MaxDD <= 0.15
                "total_alpha": 0.05,
            },
        ]
    )


def test_evaluate_backtest_pass_columns_and_maxdd_gate():
    report = evaluate_backtest(_backtest_report())
    # Per-criterion Pass columns plus overall Pass exist.
    assert {"Pass: sharpe", "Pass: max_drawdown", "Pass: total_alpha", "Pass"} <= set(
        report.columns
    )
    # Sorted by sharpe desc -> the high-Sharpe-but-failing row is first.
    assert report.iloc[0]["strategy_str"] == "2w_tp0p05_sl-0p05"
    # Exactly one strategy passes (the 2w one fails MaxDD).
    assert int(report["Pass"].sum()) == 1
    failing = report[report["strategy_str"] == "2w_tp0p05_sl-0p05"].iloc[0]
    assert failing["Pass: max_drawdown"] == False  # noqa: E712
    assert failing["Pass"] == False  # noqa: E712


def test_evaluate_backtest_best_pick_skips_higher_failing_sharpe():
    report = evaluate_backtest(_backtest_report())
    best = select_best(report, metric="sharpe")
    assert best is not None
    # Best PASSING strategy is the 1w one, even though 2w has higher Sharpe.
    assert best["strategy_str"] == "1w_tp0p05_sl-0p05"


def test_evaluate_backtest_strict_alpha_blocks_zero_alpha():
    df = pd.DataFrame(
        [
            {
                "strategy_str": "1m_tp0p05_sl-0p05",
                "sharpe": 2.0,
                "max_drawdown": 0.05,
                "total_alpha": 0.0,  # not strictly > 0
            }
        ]
    )
    report = evaluate_backtest(df)
    assert report["Pass"].iloc[0] == False  # noqa: E712
    assert select_best(report, metric="sharpe") is None


# --------------------------------------------------------------------------- #
# compare_models
# --------------------------------------------------------------------------- #
def test_compare_models_tabpfn_wins_only_when_strictly_better():
    lgbm = evaluate_strategies(_synthetic_raw_metrics())  # best Sharpe 1.5

    better = pd.DataFrame(
        [_raw_row("1w", 0.05, -0.05, 42, 1, 2.0, 0.6, 1.8, 0.08, 0.03)]
    )
    worse = pd.DataFrame(
        [_raw_row("1w", 0.05, -0.05, 42, 1, 1.0, 0.6, 1.8, 0.08, 0.03)]
    )
    assert compare_models(lgbm, evaluate_strategies(better))["winner"] == "TabPFN"
    assert compare_models(lgbm, evaluate_strategies(worse))["winner"] == "LightGBM"

    # TabPFN with nothing passing -> LightGBM stays.
    none_pass = pd.DataFrame(
        [_raw_row("1w", 0.05, -0.05, 42, 1, 0.1, 0.4, 0.9, 0.3, -0.01)]
    )
    assert compare_models(lgbm, evaluate_strategies(none_pass))["winner"] == "LightGBM"


# --------------------------------------------------------------------------- #
# paper_to_live_gate
# --------------------------------------------------------------------------- #
def test_paper_gate_recommends_when_all_clear():
    live = {
        "Portfolio Sharpe (Net)": 1.0,
        "Portfolio Max Drawdown": 0.10,
        "Win Rate": 0.60,
        "Profit Factor": 1.5,
        "Mean Alpha (Net)": 0.03,
        "Num Closed Trades": 40,
        "Track Days": 120,
        "Avg Cost (bps)": 20.0,
    }
    gate = paper_to_live_gate(live)
    assert gate["criteria_ok"] is True
    assert gate["sample_ok"] is True
    assert gate["stress_ok"] is True
    assert gate["recommend_promote"] is True


def test_paper_gate_blocks_on_small_sample():
    live = {
        "Portfolio Sharpe (Net)": 1.0,
        "Portfolio Max Drawdown": 0.10,
        "Win Rate": 0.60,
        "Profit Factor": 1.5,
        "Mean Alpha (Net)": 0.03,
        "Num Closed Trades": 5,  # too few
        "Track Days": 120,
        "Avg Cost (bps)": 20.0,
    }
    gate = paper_to_live_gate(live)
    assert gate["criteria_ok"] is True
    assert gate["sample_ok"] is False
    assert gate["recommend_promote"] is False
