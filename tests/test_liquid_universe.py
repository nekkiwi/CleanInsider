"""Unit tests for the liquid-universe pivot in src/training_pipeline.py.

Covers ModelTrainer._prepare_strategy_data in liquid mode:
  - drops rows below LIQUID_PRICE_MIN / LIQUID_ADV_MIN or with NaN adv,
  - excludes 'Price' and 'adv' from the feature matrix (filter cols, not features),
  - net-of-cost target uses the FLAT LIQUID_ROUND_TRIP_COST (not corwin_schultz),
  - legacy CS path still works when LIQUID_UNIVERSE_ONLY is False.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.training_pipeline import ModelTrainer  # noqa: E402

TARGET_COL = "alpha_1w_tp0p05_sl-0p05"


def _trainer():
    return ModelTrainer(num_folds=config.NUM_VALIDATION_FOLDS)


def _base_df():
    """4 events: 2 liquid, 1 cheap (< price min), 1 thin (< adv min / NaN)."""
    return pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CHEAP", "THIN"],
            "Filing Date": pd.to_datetime(["2024-01-02"] * 4),
            "feat_x": [1.0, 2.0, 3.0, 4.0],
            "feat_y": [0.1, 0.2, 0.3, 0.4],
            "Price": [50.0, 25.0, 3.0, 40.0],  # CHEAP below $10
            "adv": [50e6, 10e6, 20e6, np.nan],  # THIN unknown liquidity
            TARGET_COL: [0.08, -0.01, 0.20, 0.05],
            "corwin_schultz_spread": [0.01, 0.02, 0.07, 0.03],
        }
    )


def test_liquid_filter_drops_subthreshold_and_unknown(monkeypatch):
    monkeypatch.setattr(config, "LIQUID_UNIVERSE_ONLY", True)
    monkeypatch.setattr(config, "NET_OF_COST_TARGET", True)
    monkeypatch.setattr(config, "DROP_FEATURE_PREFIXES", ())

    X, y_bin, y_cont = _trainer()._prepare_strategy_data(_base_df(), TARGET_COL, 2)

    # Only AAA + BBB survive (CHEAP price<10, THIN adv NaN).
    assert len(X) == 2
    assert y_bin is not None and len(y_bin) == 2


def test_liquid_excludes_price_and_adv_from_features(monkeypatch):
    monkeypatch.setattr(config, "LIQUID_UNIVERSE_ONLY", True)
    monkeypatch.setattr(config, "NET_OF_COST_TARGET", True)
    monkeypatch.setattr(config, "DROP_FEATURE_PREFIXES", ())

    X, _, _ = _trainer()._prepare_strategy_data(_base_df(), TARGET_COL, 2)

    assert "Price" not in X.columns
    assert "adv" not in X.columns
    assert "corwin_schultz_spread" in X.columns  # still a real cost feature
    assert {"feat_x", "feat_y"} <= set(X.columns)


def test_liquid_net_target_uses_flat_cost(monkeypatch):
    monkeypatch.setattr(config, "LIQUID_UNIVERSE_ONLY", True)
    monkeypatch.setattr(config, "NET_OF_COST_TARGET", True)
    monkeypatch.setattr(config, "LIQUID_ROUND_TRIP_COST", 0.002)
    monkeypatch.setattr(config, "DROP_FEATURE_PREFIXES", ())

    df = _base_df()
    X, y_bin, y_cont = _trainer()._prepare_strategy_data(df, TARGET_COL, 2)

    # Surviving rows AAA (0.08) and BBB (-0.01); net = gross - 0.002 (FLAT),
    # NOT gross - corwin_schultz_spread (0.01 / 0.02).
    surviving = df[df["Ticker"].isin(["AAA", "BBB"])]
    expected_cont = (surviving[TARGET_COL] - 0.002).to_numpy()
    np.testing.assert_allclose(np.sort(y_cont.to_numpy()), np.sort(expected_cont))
    # AAA net 0.078 >= 0 -> 1 ; BBB net -0.012 < 0 -> 0.
    assert sorted(y_bin.tolist()) == [0, 1]


def test_legacy_cs_path_when_liquid_off(monkeypatch):
    monkeypatch.setattr(config, "LIQUID_UNIVERSE_ONLY", False)
    monkeypatch.setattr(config, "NET_OF_COST_TARGET", True)
    monkeypatch.setattr(config, "DROP_FEATURE_PREFIXES", ())

    df = _base_df()
    X, y_bin, y_cont = _trainer()._prepare_strategy_data(df, TARGET_COL, 2)

    # No liquidity filter: all 4 rows kept (all have a CS spread).
    assert len(X) == 4
    # Net uses the per-name CS spread, not the flat cost.
    expected_cont = (df[TARGET_COL] - df["corwin_schultz_spread"]).to_numpy()
    np.testing.assert_allclose(y_cont.to_numpy(), expected_cont)
    # Price/adv still excluded from features in legacy mode too.
    assert "Price" not in X.columns and "adv" not in X.columns


def test_liquid_filter_all_dropped_returns_none(monkeypatch):
    monkeypatch.setattr(config, "LIQUID_UNIVERSE_ONLY", True)
    monkeypatch.setattr(config, "NET_OF_COST_TARGET", True)

    df = _base_df()
    df["adv"] = np.nan  # nobody has known liquidity
    X, y_bin, y_cont = _trainer()._prepare_strategy_data(df, TARGET_COL, 2)
    assert X is None and y_bin is None and y_cont is None


def test_attach_liquidity_columns_guards_missing_sources(monkeypatch, tmp_path):
    # Point config at non-existent paths -> Price/adv filled with NaN, no crash.
    monkeypatch.setattr(config, "MASTER_EVENT_LIST_PATH", tmp_path / "nope.parquet")
    monkeypatch.setattr(config, "ADV_COMPONENT_PATH", tmp_path / "nope_adv.parquet")

    merged = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "Filing Date": pd.to_datetime(["2024-01-02"]),
            "feat_x": [1.0],
            TARGET_COL: [0.05],
        }
    )
    out = _trainer()._attach_liquidity_columns(merged)
    assert "Price" in out.columns and "adv" in out.columns
    assert out["Price"].isna().all()
    assert out["adv"].isna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
