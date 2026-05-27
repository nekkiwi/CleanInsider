# tests/test_sizing.py
"""
Stage 4: volatility-targeted position sizing tests.

Covers:
- get_atr on AlpacaTradingClient (mocked data client, no network)
- calculate_vol_target_sizes: hand-computed shares/dollars, 5% per-name cap,
  batch scaling to TARGET_NET_EXPOSURE (respecting current_exposure_dollars),
  ATR-missing pct-stop fallback (finite shares, never inf)
- apply_kelly_cap no-op pass-through
- size_positions(method="minmax") parity with the legacy sizing path
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.alpaca.position_sizer import PositionSizer
from src.alpaca.trading_client import AlpacaTradingClient


# --------------------------------------------------------------------------- #
# get_atr
# --------------------------------------------------------------------------- #
def _make_bars(highs, lows, closes):
    """Build a list of objects mimicking alpaca-py Bar attributes."""
    bars = []
    for h, low, c in zip(highs, lows, closes):
        bar = MagicMock()
        bar.high = h
        bar.low = low
        bar.close = c
        bars.append(bar)
    return bars


class TestGetATR:
    def _client_with_data(self, data_map):
        """An AlpacaTradingClient with client/data_client injected (no network)."""
        client = AlpacaTradingClient.__new__(AlpacaTradingClient)
        client.client = MagicMock()  # marks "connected"
        data_client = MagicMock()

        barset = MagicMock()
        barset.data = data_map  # {symbol: [bar, ...]}
        data_client.get_stock_bars.return_value = barset

        client.data_client = data_client
        return client

    def test_atr_known_values(self):
        # 4 bars -> 3 true ranges. Make TR a constant 2.0 each step so
        # ATR(period that yields 3 TRs) = 2.0 regardless of averaging window.
        # high-low = 2.0 each bar, and gaps small so TR = high-low.
        highs = [12, 12, 12, 12]
        lows = [10, 10, 10, 10]
        closes = [11, 11, 11, 11]
        client = self._client_with_data({"AAA": _make_bars(highs, lows, closes)})

        atr = client.get_atr(["AAA"], period=3)
        assert pytest.approx(atr["AAA"], rel=1e-9) == 2.0

    def test_atr_true_range_uses_prev_close(self):
        # Gap up: prev close 10, next bar high 20 low 15 -> TR = max(5, |20-10|, |15-10|) = 10
        # Build 2 bars: first close=10, second high=20 low=15 close=18.
        b0 = MagicMock()
        b0.high, b0.low, b0.close = 11, 9, 10
        b1 = MagicMock()
        b1.high, b1.low, b1.close = 20, 15, 18
        client = self._client_with_data({"BBB": [b0, b1]})

        atr = client.get_atr(["BBB"], period=1)
        # single TR = max(20-15, |20-10|, |15-10|) = 10
        assert pytest.approx(atr["BBB"], rel=1e-9) == 10.0

    def test_atr_missing_symbol_omitted(self):
        client = self._client_with_data({})  # no data for anything
        atr = client.get_atr(["MISSING"], period=3)
        assert "MISSING" not in atr
        assert atr == {}

    def test_atr_insufficient_bars_omitted(self):
        # only 1 bar -> cannot form a true range -> omit
        b0 = MagicMock()
        b0.high, b0.low, b0.close = 11, 9, 10
        client = self._client_with_data({"CCC": [b0]})
        atr = client.get_atr(["CCC"], period=3)
        assert "CCC" not in atr

    def test_atr_no_client_returns_empty(self):
        client = AlpacaTradingClient.__new__(AlpacaTradingClient)
        client.client = None
        atr = client.get_atr(["AAA"])
        assert atr == {}


# --------------------------------------------------------------------------- #
# calculate_vol_target_sizes
# --------------------------------------------------------------------------- #
class TestVolTargetSizing:
    def test_known_atr_shares_and_dollars(self):
        sizer = PositionSizer(max_position_size=0.05)
        pv = 100_000.0
        # risk_budget = pv * 0.0125 = 1250
        # ATR = 2.0, mult = 1.5 -> stop_distance = 3.0
        # shares = 1250 / 3.0 = 416.666..., base_dollar = shares * price(50) = 20833.33
        # Single name -> conviction midpoint 0.625, no spread -> haircut 1.0
        # dollar = 20833.33 * 0.625 = 13020.83 -> capped at 5% * pv = 5000
        predicted_returns = pd.Series({"AAA": 0.10})
        prices = pd.Series({"AAA": 50.0})
        atr_map = {"AAA": 2.0}

        result = sizer.calculate_vol_target_sizes(
            predicted_returns,
            prices,
            atr_map,
            strategy_sl=-0.05,
            portfolio_value=pv,
        )
        dollars = result["dollars"]
        shares = result["shares"]

        # capped at MAX_POSITION_SIZE * pv = 5000
        assert pytest.approx(dollars["AAA"], rel=1e-9) == 5000.0
        # shares = floor(5000 / 50) = 100
        assert shares["AAA"] == 100

    def test_per_name_cap_clamps_high_conviction(self):
        # Two names, one very high conviction. ATR small so base_dollar is huge.
        # Verify the high-conviction name is clamped to 5% * pv.
        sizer = PositionSizer(max_position_size=0.05)
        pv = 100_000.0
        predicted_returns = pd.Series({"HIGH": 0.50, "LOW": 0.01})
        prices = pd.Series({"HIGH": 10.0, "LOW": 10.0})
        atr_map = {"HIGH": 0.10, "LOW": 0.10}  # tiny ATR -> huge base size

        result = sizer.calculate_vol_target_sizes(
            predicted_returns, prices, atr_map, strategy_sl=-0.05, portfolio_value=pv
        )
        dollars = result["dollars"]
        cap = 0.05 * pv
        assert dollars["HIGH"] <= cap + 1e-6
        assert pytest.approx(dollars["HIGH"], rel=1e-9) == cap

    def test_batch_scales_to_target_net_exposure(self):
        # Many names each capped at 5% -> sum exceeds 50% of pv -> scale down to 50%.
        sizer = PositionSizer(max_position_size=0.05)
        pv = 100_000.0
        names = [f"T{i}" for i in range(20)]  # 20 * 5% = 100% > 50%
        predicted_returns = pd.Series({n: 0.10 for n in names})
        prices = pd.Series({n: 10.0 for n in names})
        atr_map = {n: 0.01 for n in names}  # tiny ATR -> each hits the cap

        result = sizer.calculate_vol_target_sizes(
            predicted_returns, prices, atr_map, strategy_sl=-0.05, portfolio_value=pv
        )
        total = result["dollars"].sum()
        assert pytest.approx(total, rel=1e-9) == config.TARGET_NET_EXPOSURE * pv

    def test_batch_respects_current_exposure(self):
        # Already 30% exposed; target 50% -> only 20% of pv of new exposure allowed.
        sizer = PositionSizer(max_position_size=0.05)
        pv = 100_000.0
        names = [f"T{i}" for i in range(20)]
        predicted_returns = pd.Series({n: 0.10 for n in names})
        prices = pd.Series({n: 10.0 for n in names})
        atr_map = {n: 0.01 for n in names}

        result = sizer.calculate_vol_target_sizes(
            predicted_returns,
            prices,
            atr_map,
            strategy_sl=-0.05,
            portfolio_value=pv,
            current_exposure_dollars=30_000.0,
        )
        new_total = result["dollars"].sum()
        target_new = config.TARGET_NET_EXPOSURE * pv - 30_000.0  # 20000
        assert pytest.approx(new_total, rel=1e-9) == target_new

    def test_atr_missing_pct_stop_fallback_finite(self):
        # No ATR -> fallback stop = abs(strategy_sl) * price.
        # strategy_sl=-0.05, price=100 -> stop_distance = 5.0
        # risk_budget=1250 -> shares = 250, base_dollar = 250*100 = 25000 -> cap 5000
        sizer = PositionSizer(max_position_size=0.05)
        pv = 100_000.0
        predicted_returns = pd.Series({"NOATR": 0.10})
        prices = pd.Series({"NOATR": 100.0})
        atr_map = {}  # missing

        result = sizer.calculate_vol_target_sizes(
            predicted_returns, prices, atr_map, strategy_sl=-0.05, portfolio_value=pv
        )
        dollars = result["dollars"]
        shares = result["shares"]
        assert np.isfinite(dollars["NOATR"])
        assert np.isfinite(shares["NOATR"])
        assert shares["NOATR"] >= 0
        # capped at 5000
        assert pytest.approx(dollars["NOATR"], rel=1e-9) == 5000.0

    def test_atr_zero_uses_fallback(self):
        # ATR present but 0 -> stop_distance would be 0 -> must fall back to pct stop.
        sizer = PositionSizer(max_position_size=0.05)
        pv = 100_000.0
        predicted_returns = pd.Series({"ZERO": 0.10})
        prices = pd.Series({"ZERO": 100.0})
        atr_map = {"ZERO": 0.0}

        result = sizer.calculate_vol_target_sizes(
            predicted_returns, prices, atr_map, strategy_sl=-0.05, portfolio_value=pv
        )
        assert np.isfinite(result["dollars"]["ZERO"])
        assert np.isfinite(result["shares"]["ZERO"])

    def test_spread_haircut_applied(self):
        # A wide spread should reduce the dollar allocation vs no-spread.
        sizer = PositionSizer(max_position_size=0.05, max_spread_cost=0.03)
        pv = 100_000.0
        predicted_returns = pd.Series({"A": 0.10, "B": 0.10})
        prices = pd.Series({"A": 100.0, "B": 100.0})
        # large ATR so base_dollar is small and NOT capped (so haircut is visible)
        atr_map = {"A": 50.0, "B": 50.0}
        spreads = pd.Series({"A": 0.0002, "B": 0.04})  # B above max_spread_cost

        result = sizer.calculate_vol_target_sizes(
            predicted_returns,
            prices,
            atr_map,
            strategy_sl=-0.05,
            portfolio_value=pv,
            spreads=spreads,
        )
        dollars = result["dollars"]
        # B's half-spread = 0.02 > max_spread_cost 0.03? half = 0.02 < 0.03 actually.
        # Use formula: haircut = ref/half capped at 1; ref=0.005, half=0.02 -> 0.25
        # A: half=0.0001 -> haircut 1.0. So B < A.
        assert dollars["B"] < dollars["A"]

    def test_empty_input(self):
        sizer = PositionSizer()
        result = sizer.calculate_vol_target_sizes(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            {},
            strategy_sl=-0.05,
            portfolio_value=100_000.0,
        )
        assert result["dollars"].empty
        assert result["shares"].empty


# --------------------------------------------------------------------------- #
# apply_kelly_cap
# --------------------------------------------------------------------------- #
class TestKellyCap:
    def test_kelly_is_noop(self):
        sizer = PositionSizer()
        dollars = pd.Series({"A": 1000.0, "B": 2000.0})
        out = sizer.apply_kelly_cap(dollars)
        pd.testing.assert_series_equal(out, dollars)


# --------------------------------------------------------------------------- #
# minmax parity
# --------------------------------------------------------------------------- #
class TestMinmaxParity:
    """size_positions(method='minmax') must equal the legacy default behavior."""

    def _legacy_size_positions(
        self, sizer, signals_df, portfolio_value, current, spreads
    ):
        """Inlined copy of the PRE-refactor size_positions body for parity check."""
        from typing import Dict  # noqa: F401

        df = signals_df.copy()
        current_exposure = sum((current or {}).values())

        if "position_size" in df.columns:
            original_sizes = df.set_index("Ticker")["position_size"]
        elif "predicted_return" in df.columns:
            original_sizes = sizer.calculate_base_sizes(
                df.set_index("Ticker")["predicted_return"]
            )
        else:
            original_sizes = pd.Series(0.5, index=df["Ticker"])

        base_sizes = original_sizes.copy()
        spread_haircuts = pd.Series(1.0, index=df["Ticker"])

        if spreads is not None:
            if isinstance(spreads, dict):
                spreads = pd.Series(spreads)
            base_sizes = sizer.apply_spread_haircut(original_sizes, spreads)
            for t in df["Ticker"]:
                orig = original_sizes.get(t, 1.0)
                final = base_sizes.get(t, 0.0)
                spread_haircuts[t] = final / max(orig, 0.001) if orig > 0 else 0

        dollar_sizes = sizer.calculate_dollar_sizes(
            base_sizes, portfolio_value, current_exposure
        )
        prices = (
            df.set_index("Ticker")["Price"] if "Price" in df.columns else pd.Series()
        )
        if not prices.empty:
            shares = sizer.calculate_shares(dollar_sizes, prices)
        else:
            shares = pd.Series(dtype=int)

        cols = ["Ticker"]
        if "Filing Date" in df.columns:
            cols.append("Filing Date")
        result = df[cols].copy()
        result["base_size"] = base_sizes.reindex(df["Ticker"]).values
        result["dollar_size"] = dollar_sizes.reindex(df["Ticker"]).values
        result["shares"] = (
            shares.reindex(df["Ticker"]).values if not shares.empty else 0
        )
        result["spread_haircut"] = spread_haircuts.reindex(df["Ticker"]).values
        if "Price" in df.columns:
            result["price"] = df["Price"].values
        if "predicted_return" in df.columns:
            result["predicted_return"] = df["predicted_return"].values
        if "confidence" in df.columns:
            result["confidence"] = df["confidence"].values
        result = result[result["dollar_size"] > 0].copy()
        return result.sort_values("dollar_size", ascending=False)

    def _signals(self):
        return pd.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT", "GOOGL"],
                "predicted_return": [0.10, 0.05, 0.02],
                "Price": [150.0, 300.0, 120.0],
            }
        )

    def test_minmax_parity_no_spread(self):
        sizer = PositionSizer(
            max_position_size=0.05, max_total_exposure=0.50, min_position_dollars=100
        )
        signals = self._signals()
        pv = 100_000.0
        current = {}

        expected = self._legacy_size_positions(sizer, signals, pv, current, None)
        got = sizer.size_positions(signals, pv, current, method="minmax")

        pd.testing.assert_frame_equal(
            got.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_minmax_parity_with_spread(self):
        sizer = PositionSizer(
            max_position_size=0.05, max_total_exposure=0.50, min_position_dollars=100
        )
        signals = self._signals()
        pv = 100_000.0
        current = {"EXISTING": 10_000.0}
        spreads = pd.Series({"AAPL": 0.002, "MSFT": 0.010, "GOOGL": 0.025})

        expected = self._legacy_size_positions(sizer, signals, pv, current, spreads)
        got = sizer.size_positions(
            signals, pv, current, spreads=spreads, method="minmax"
        )

        pd.testing.assert_frame_equal(
            got.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_minmax_parity_with_position_size_column(self):
        sizer = PositionSizer(
            max_position_size=0.05, max_total_exposure=0.50, min_position_dollars=100
        )
        signals = pd.DataFrame(
            {
                "Ticker": ["A", "B", "C"],
                "position_size": [1.0, 0.5, 0.25],
                "predicted_return": [0.10, 0.05, 0.02],
                "Price": [50.0, 40.0, 30.0],
            }
        )
        pv = 100_000.0
        current = {}

        expected = self._legacy_size_positions(sizer, signals, pv, current, None)
        got = sizer.size_positions(signals, pv, current, method="minmax")

        pd.testing.assert_frame_equal(
            got.reset_index(drop=True), expected.reset_index(drop=True)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
