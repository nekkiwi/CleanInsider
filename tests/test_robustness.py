"""
Unit tests for src/training/robustness.py.

Everything is synthetic and injected (stub models, OHLCV, SPX, feature frames),
mirroring tests/test_ensemble_backtest.py — no disk, no trained pickles.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.ensemble_backtest import strategy_string  # noqa: E402
from src.training.robustness import (  # noqa: E402
    deflated_sharpe,
    sweep_capital,
    sweep_cost,
    sweep_entry_timing,
    sweep_liquidity,
    walk_forward_oos,
)

BDAYS = pd.bdate_range("2024-01-01", periods=60)


# --------------------------------------------------------------------------- #
# Stubs (same shapes as test_ensemble_backtest)
# --------------------------------------------------------------------------- #
class _Classifier:
    def __init__(self, votes):
        self._votes = np.asarray(votes, dtype=int)

    def predict(self, X):
        return self._votes[: len(X)]


class _Regressor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def predict(self, X):
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
    def loader(model_dir: Path):
        seed = int(model_dir.name.split("_")[1])
        fold = int(model_dir.parent.name.split("_")[1])
        return models[(fold, seed)]

    return loader


def _build_inputs(tmp_path):
    """3 stub models on fold 1; AAA + BBB are strong buys, CCC/DDD no-buy."""
    strategy = ("1w", 0.05, -0.05)
    folds, seeds = [1], [42, 123, 2024]

    base = tmp_path / "models"
    strat_dir = base / strategy_string(strategy)
    for f in folds:
        for s in seeds:
            (strat_dir / f"fold_{f}" / f"seed_{s}").mkdir(parents=True, exist_ok=True)

    features = ["f0", "f1"]
    test_df = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC", "DDD"],
            "Filing Date": [BDAYS[0], BDAYS[0], BDAYS[0], BDAYS[0]],
            "f0": [1.0, 2.0, 3.0, 4.0],
            "f1": [0.5, 0.6, 0.7, 0.8],
            "Price": [50.0, 50.0, 50.0, 50.0],
            "adv": [50e6, 50e6, 50e6, 50e6],
        }
    )
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
    # Non-constant up-drift paths (so a shifted entry window changes the realized
    # return) that still trend UP overall (so a positive entry cost reduces net
    # alpha); no TP/SL hit over a 30-bar window at +/-50% thresholds.
    rng = np.random.default_rng(7)
    aaa = 100.0 * np.cumprod(np.r_[1.0, 1.0 + 0.004 + 0.01 * rng.standard_normal(29)])
    bbb = 100.0 * np.cumprod(np.r_[1.0, 1.0 + 0.003 + 0.01 * rng.standard_normal(29)])
    ohlcv = {"AAA": _ohlcv(BDAYS[:30], aaa), "BBB": _ohlcv(BDAYS[:30], bbb)}
    return strategy, folds, seeds, base, test_df, models, ohlcv


def _common_kwargs(folds, seeds, base, test_df, models, ohlcv):
    return dict(
        models_base_path=base,
        test_features_df=test_df,
        ohlcv_loader=_make_ohlcv_loader(ohlcv),
        spx_arrays=_flat_spx(BDAYS[:30]),
        folds=folds,
        seeds=seeds,
        model_loader=_make_model_loader(models),
        vote_threshold=0.5,
    )


# --------------------------------------------------------------------------- #
# Entry timing
# --------------------------------------------------------------------------- #
def test_sweep_entry_timing_one_row_per_offset_and_shifts(tmp_path):
    strategy, folds, seeds, base, test_df, models, ohlcv = _build_inputs(tmp_path)
    kw = _common_kwargs(folds, seeds, base, test_df, models, ohlcv)
    df = sweep_entry_timing(strategy, offsets=(0, 1, 2), liquid_mode=True, **kw)

    assert list(df["entry_offset"]) == [0, 1, 2]
    assert len(df) == 3
    # The 3 offsets enter at different prices on an upward drift -> total alpha
    # must not be identical across all three.
    assert df["total_alpha"].nunique() >= 2


def test_sweep_entry_timing_offset_zero_matches_baseline(tmp_path):
    from src.training.ensemble_backtest import backtest_strategy

    strategy, folds, seeds, base, test_df, models, ohlcv = _build_inputs(tmp_path)
    kw = _common_kwargs(folds, seeds, base, test_df, models, ohlcv)
    df = sweep_entry_timing(strategy, offsets=(0,), liquid_mode=True, **kw)
    baseline = backtest_strategy(strategy, liquid_mode=True, **kw)
    assert np.isclose(
        df.iloc[0]["total_alpha"], baseline["total_alpha"], equal_nan=True
    )


# --------------------------------------------------------------------------- #
# Cost sweep
# --------------------------------------------------------------------------- #
def test_sweep_cost_monotonic_and_break_even(tmp_path):
    strategy, folds, seeds, base, test_df, models, ohlcv = _build_inputs(tmp_path)
    kw = _common_kwargs(folds, seeds, base, test_df, models, ohlcv)
    df = sweep_cost(strategy, cost_bps=(0, 20, 60, 200, 1000), **kw)

    assert list(df["cost_bps"]) == [0.0, 20.0, 60.0, 200.0, 1000.0]
    # Higher cost -> lower (or equal) net total alpha (monotone non-increasing).
    # (Sharpe is mean/std; a first-day cost subtraction need NOT be monotone in
    # Sharpe once alpha crosses zero, so only total_alpha is asserted monotone.)
    alpha = df["total_alpha"].to_numpy()
    assert np.all(np.diff(alpha) <= 1e-12)
    # With a 1000 bps round-trip cost the edge is destroyed -> a break-even flag.
    assert df["break_even"].any()
    # The flagged row is the FIRST with total_alpha <= 0.
    be_idx = df.index[df["break_even"]][0]
    assert df.loc[be_idx, "total_alpha"] <= 0


def test_sweep_cost_zero_cost_is_best(tmp_path):
    strategy, folds, seeds, base, test_df, models, ohlcv = _build_inputs(tmp_path)
    kw = _common_kwargs(folds, seeds, base, test_df, models, ohlcv)
    df = sweep_cost(strategy, cost_bps=(0, 50), **kw)
    assert df.iloc[0]["total_alpha"] >= df.iloc[1]["total_alpha"]


# --------------------------------------------------------------------------- #
# Liquidity sweep
# --------------------------------------------------------------------------- #
def test_sweep_liquidity_tiers_filter_universe(tmp_path):
    strategy, folds, seeds, base, test_df, models, ohlcv = _build_inputs(tmp_path)
    # AAA price 50 adv 50M ; BBB price 50 adv 8M. A $10M ADV floor drops BBB.
    test_df = test_df.copy()
    test_df["adv"] = [50e6, 8e6, 8e6, 8e6]
    kw = _common_kwargs(folds, seeds, base, test_df, models, ohlcv)
    df = sweep_liquidity(strategy, tiers=[(10, 5e6), (10, 10e6)], **kw)

    assert list(df["price_min"]) == [10.0, 10.0]
    assert list(df["adv_min"]) == [5e6, 10e6]
    # Loose tier keeps both buys; tight ADV tier (10M) keeps only AAA.
    assert df.iloc[0]["n_trades"] == 2
    assert df.iloc[1]["n_trades"] == 1


# --------------------------------------------------------------------------- #
# Capital sweep
# --------------------------------------------------------------------------- #
def test_sweep_capital_cartesian_product(tmp_path):
    strategy, folds, seeds, base, test_df, models, ohlcv = _build_inputs(tmp_path)
    kw = _common_kwargs(folds, seeds, base, test_df, models, ohlcv)
    df = sweep_capital(
        strategy, per_name=(0.025, 0.05), gross=(0.5, 1.0), liquid_mode=True, **kw
    )
    assert len(df) == 4  # 2 x 2 grid
    assert set(df["per_name_cap"]) == {0.025, 0.05}
    assert set(df["max_gross_exposure"]) == {0.5, 1.0}
    assert df["n_trades"].iloc[0] == 2


# --------------------------------------------------------------------------- #
# Deflated Sharpe
# --------------------------------------------------------------------------- #
def test_deflated_sharpe_strong_signal_high_prob():
    """A clearly positive, long, low-noise return series -> high Prob(SR>0)."""
    rng = np.random.default_rng(0)
    # mean 0.001/day, std 0.005 -> daily SR ~0.2, very significant over 1000 days
    r = 0.001 + 0.005 * rng.standard_normal(1000)
    dsr = deflated_sharpe(r, n_trials=10)
    assert 0.9 <= dsr <= 1.0


def test_deflated_sharpe_zero_mean_low_prob():
    """A pure-noise zero-mean series -> Prob(SR>0) near 0.5 or below."""
    rng = np.random.default_rng(1)
    r = 0.005 * rng.standard_normal(1000)
    dsr = deflated_sharpe(r, n_trials=50)
    assert dsr < 0.7


def test_deflated_sharpe_more_trials_lowers_prob():
    """More trials raise the benchmark -> a fixed series gets a lower DSR."""
    rng = np.random.default_rng(2)
    r = 0.0006 + 0.005 * rng.standard_normal(800)
    few = deflated_sharpe(r, n_trials=1)
    many = deflated_sharpe(r, n_trials=500)
    assert few >= many


def test_deflated_sharpe_accepts_sharpe_array():
    """Passing an array of per-trial annualized Sharpes also works."""
    sharpes = np.array([0.5, 1.0, 1.5, 4.72, 2.0])
    dsr = deflated_sharpe(sharpes, n_trials=5)
    assert 0.0 <= dsr <= 1.0


def test_deflated_sharpe_too_short_is_nan():
    assert np.isnan(deflated_sharpe([0.01], n_trials=3))


# --------------------------------------------------------------------------- #
# Walk-forward OOS
# --------------------------------------------------------------------------- #
def _build_wfo_inputs(tmp_path):
    """One synthetic fold with a validation_data.parquet on disk + stub models."""
    strategy = ("1w", 0.05, -0.05)
    seeds = [42, 123, 2024]
    fold = 1

    base = tmp_path / "models"
    strat_dir = base / strategy_string(strategy)
    for s in seeds:
        (strat_dir / f"fold_{fold}" / f"seed_{s}").mkdir(parents=True, exist_ok=True)

    features = ["f0", "f1"]
    feats = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC"],
            "Filing Date": [BDAYS[0], BDAYS[0], BDAYS[0]],
            "f0": [1.0, 2.0, 3.0],
            "f1": [0.5, 0.6, 0.7],
            "Price": [50.0, 50.0, 50.0],
            "adv": [50e6, 50e6, 50e6],
        }
    )
    features_dir = tmp_path / "features"
    fold_dir = features_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(fold_dir / "validation_data.parquet")

    targets_dir = tmp_path / "targets"  # unused by the simulation path but real
    (targets_dir / f"fold_{fold}").mkdir(parents=True, exist_ok=True)

    models = {
        (1, 42): (
            _Classifier([1, 1, 0]),
            _Regressor([0.2, 0.1, 0.0]),
            _metadata(features),
        ),
        (1, 123): (
            _Classifier([1, 1, 0]),
            _Regressor([0.18, 0.12, 0.0]),
            _metadata(features),
        ),
        (1, 2024): (
            _Classifier([1, 1, 0]),
            _Regressor([0.22, 0.08, 0.0]),
            _metadata(features),
        ),
    }
    aaa = 100.0 * np.cumprod(np.r_[1.0, np.full(29, 1.005)])
    bbb = 100.0 * np.cumprod(np.r_[1.0, np.full(29, 1.004)])
    ohlcv = {"AAA": _ohlcv(BDAYS[:30], aaa), "BBB": _ohlcv(BDAYS[:30], bbb)}
    return strategy, [fold], seeds, base, features_dir, targets_dir, models, ohlcv


def test_walk_forward_oos_one_fold_aggregates(tmp_path):
    strategy, folds, seeds, base, features_dir, targets_dir, models, ohlcv = (
        _build_wfo_inputs(tmp_path)
    )
    df = walk_forward_oos(
        strategy,
        models_base_path=base,
        features_dir=features_dir,
        targets_dir=targets_dir,
        spx_arrays=_flat_spx(BDAYS[:30]),
        ohlcv_loader=_make_ohlcv_loader(ohlcv),
        folds=folds,
        seeds=seeds,
        model_loader=_make_model_loader(models),
        vote_threshold=0.5,
        liquid_round_trip_cost=0.002,
        price_min=10.0,
        adv_min=5e6,
    )
    # One per-fold row + an OVERALL row.
    assert "OVERALL" in set(df["fold"].astype(str))
    fold_row = df[df["fold"].astype(str) == "1"].iloc[0]
    assert fold_row["n_trades"] == 2  # AAA + BBB clear the vote + liquidity
    assert fold_row["n_days"] > 0
    assert np.isfinite(fold_row["sharpe"])


def test_walk_forward_oos_liquidity_filter_drops_illiquid(tmp_path):
    strategy, folds, seeds, base, features_dir, targets_dir, models, ohlcv = (
        _build_wfo_inputs(tmp_path)
    )
    # Re-write the validation frame with BBB below the $10 price floor.
    feats = pd.read_parquet(features_dir / "fold_1" / "validation_data.parquet")
    feats["Price"] = [50.0, 8.0, 50.0]
    feats.to_parquet(features_dir / "fold_1" / "validation_data.parquet")
    df = walk_forward_oos(
        strategy,
        models_base_path=base,
        features_dir=features_dir,
        targets_dir=targets_dir,
        spx_arrays=_flat_spx(BDAYS[:30]),
        ohlcv_loader=_make_ohlcv_loader(ohlcv),
        folds=folds,
        seeds=seeds,
        model_loader=_make_model_loader(models),
        vote_threshold=0.5,
        price_min=10.0,
        adv_min=5e6,
    )
    fold_row = df[df["fold"].astype(str) == "1"].iloc[0]
    assert fold_row["n_trades"] == 1  # only AAA survives the price floor
