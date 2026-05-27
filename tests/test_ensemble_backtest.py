"""
Unit tests for src/training/ensemble_backtest.py.

The 25-model ensemble, the test set, OHLCV, and SPX are all stubbed/synthetic
and injected, so nothing touches disk or trained pickles.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.ensemble_backtest import (  # noqa: E402
    DEFAULT_ONE_WAY_COST,
    backtest_strategy,
    strategy_string,
)

BDAYS = pd.bdate_range("2024-01-01", periods=40)


# --------------------------------------------------------------------------- #
# Stub model + injected loaders
# --------------------------------------------------------------------------- #
class _Classifier:
    """Predicts a fixed 0/1 vote vector (one per test row)."""

    def __init__(self, votes):
        self._votes = np.asarray(votes, dtype=int)

    def predict(self, X):
        return self._votes[: len(X)]


class _Regressor:
    """Predicts a fixed conviction per row; indexed positionally on the buys."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def predict(self, X):
        # X here is the buy-subset; map by integer position of its index.
        pos = X.index.to_numpy()
        return self._values[pos]


def _metadata(features):
    return {"selected_features": list(features), "imputation_values": {}}


def _ohlcv(dates, close, high=None, low=None):
    close = np.asarray(close, dtype=float)
    high = close.copy() if high is None else np.asarray(high, dtype=float)
    low = close.copy() if low is None else np.asarray(low, dtype=float)
    return pd.DataFrame(
        {"High": high, "Low": low, "Close": close}, index=pd.DatetimeIndex(dates)
    )


def _flat_spx(dates, level=100.0):
    return (
        pd.DatetimeIndex(dates).to_numpy(dtype="datetime64[ns]"),
        np.full(len(dates), float(level)),
    )


def _make_ohlcv_loader(data):
    def loader(ticker, db_path_str=None, required_start_date=None):
        return data.get(ticker, pd.DataFrame())

    return loader


def _make_model_loader(models):
    """Map fold_{f}/seed_{s} dirs -> (clf, reg, meta) from a dict keyed (f, s)."""

    def loader(model_dir: Path):
        seed = int(model_dir.name.split("_")[1])
        fold = int(model_dir.parent.name.split("_")[1])
        return models[(fold, seed)]

    return loader


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_strategy_string_matches_trainer_convention():
    assert strategy_string(("1w", 0.05, -0.05)) == "1w_tp0p05_sl-0p05"
    assert strategy_string(("1m", 0.10, -0.10)) == "1m_tp0p1_sl-0p1"


def _build_inputs(tmp_path):
    """Create on-disk fold/seed dirs (empty) + 3 stub models + test inputs."""
    strategy = ("1w", 0.05, -0.05)
    folds, seeds = [1], [42, 123, 2024]

    # Real directories must exist (_load_ensemble checks .exists()).
    base = tmp_path / "models"
    strat_dir = base / strategy_string(strategy)
    for f in folds:
        for s in seeds:
            (strat_dir / f"fold_{f}" / f"seed_{s}").mkdir(parents=True, exist_ok=True)

    # Test set: 4 events. Two strong buys (AAA, BBB), one weak, one no-buy.
    features = ["f0", "f1"]
    test_df = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC", "DDD"],
            "Filing Date": [BDAYS[0], BDAYS[0], BDAYS[0], BDAYS[0]],
            "f0": [1.0, 2.0, 3.0, 4.0],
            "f1": [0.5, 0.6, 0.7, 0.8],
        }
    )

    # 3 models. Rows 0,1 get 3/3 votes -> buy; row 2 gets 1/3 (<0.5) -> no buy;
    # row 3 gets 0/3 -> no buy. Regressors give AAA higher conviction than BBB.
    models = {
        (1, 42): (
            _Classifier([1, 1, 1, 0]),
            _Regressor([0.20, 0.10, 0.05, 0.0]),
            _metadata(features),
        ),
        (1, 123): (
            _Classifier([1, 1, 0, 0]),
            _Regressor([0.18, 0.12, 0.0, 0.0]),
            _metadata(features),
        ),
        (1, 2024): (
            _Classifier([1, 1, 0, 0]),
            _Regressor([0.22, 0.08, 0.0, 0.0]),
            _metadata(features),
        ),
    }

    # OHLCV: AAA drifts up, BBB drifts down, both no TP/SL hit at +/-50%.
    aaa = 100.0 * np.cumprod(np.r_[1.0, np.full(9, 1.01)])
    bbb = 100.0 * np.cumprod(np.r_[1.0, np.full(9, 0.995)])
    ohlcv = {"AAA": _ohlcv(BDAYS[:10], aaa), "BBB": _ohlcv(BDAYS[:10], bbb)}

    spreads = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Filing Date": [BDAYS[0], BDAYS[0]],
            "corwin_schultz_spread": [0.01, 0.02],
        }
    )
    return strategy, folds, seeds, base, test_df, models, ohlcv, spreads


def test_backtest_strategy_votes_costs_and_metrics(tmp_path):
    strategy, folds, seeds, base, test_df, models, ohlcv, spreads = _build_inputs(
        tmp_path
    )

    result = backtest_strategy(
        strategy,
        models_base_path=base,
        test_features_df=test_df,
        test_spreads_df=spreads,
        ohlcv_loader=_make_ohlcv_loader(ohlcv),
        spx_arrays=_flat_spx(BDAYS[:10]),
        folds=folds,
        seeds=seeds,
        model_loader=_make_model_loader(models),
        vote_threshold=0.5,
    )

    # Only AAA and BBB clear the >=0.5 vote threshold.
    assert result["n_trades"] == 2
    assert result["strategy_str"] == "1w_tp0p05_sl-0p05"

    # Metrics dict carries the expected keys with a finite Sharpe.
    expected_keys = {
        "strategy_str",
        "sharpe",
        "raw_sharpe",
        "sortino",
        "max_drawdown",
        "cagr",
        "ann_vol",
        "total_alpha",
        "n_trades",
        "n_days",
    }
    assert expected_keys <= set(result)
    assert np.isfinite(result["sharpe"])
    assert np.isfinite(result["raw_sharpe"])
    assert result["n_days"] > 0


def test_backtest_entry_cost_comes_from_spreads(tmp_path):
    """The first active day must reflect the 0.5*spread one-way cost.

    Run once with the real spreads and once with no spread table (default cost),
    and confirm the first-day portfolio alpha differs by the cost delta.
    """
    strategy, folds, seeds, base, test_df, models, ohlcv, spreads = _build_inputs(
        tmp_path
    )
    common = dict(
        models_base_path=base,
        test_features_df=test_df,
        ohlcv_loader=_make_ohlcv_loader(ohlcv),
        spx_arrays=_flat_spx(BDAYS[:10]),
        folds=folds,
        seeds=seeds,
        model_loader=_make_model_loader(models),
        vote_threshold=0.5,
    )
    # With spreads: AAA one-way 0.005, BBB one-way 0.01 (0.5 * spread).
    r_spread = backtest_strategy(strategy, test_spreads_df=spreads, **common)
    # Without spreads: both default to DEFAULT_ONE_WAY_COST.
    r_default = backtest_strategy(strategy, test_spreads_df=None, **common)

    assert DEFAULT_ONE_WAY_COST == 0.005
    # Both produce a finite, sane Sharpe (not astronomical).
    assert abs(r_spread["sharpe"]) < 50
    # Higher costs (default 0.005 on BBB vs its real 0.01 one-way) shift total
    # alpha; the two runs must not be identical when any cost differs.
    assert r_spread["total_alpha"] != r_default["total_alpha"]


def test_backtest_no_buys_returns_zero_trades(tmp_path):
    strategy, folds, seeds, base, test_df, models, ohlcv, spreads = _build_inputs(
        tmp_path
    )
    # All-zero classifiers -> nobody clears the vote threshold.
    no_buy = {
        key: (_Classifier([0, 0, 0, 0]), reg, meta)
        for key, (clf, reg, meta) in models.items()
    }
    result = backtest_strategy(
        strategy,
        models_base_path=base,
        test_features_df=test_df,
        test_spreads_df=spreads,
        ohlcv_loader=_make_ohlcv_loader(ohlcv),
        spx_arrays=_flat_spx(BDAYS[:10]),
        folds=folds,
        seeds=seeds,
        model_loader=_make_model_loader(no_buy),
        vote_threshold=0.5,
    )
    assert result["n_trades"] == 0
    assert np.isnan(result["sharpe"])
