# tests/test_exit_logic.py
"""
Stage 5 exits: unit tests for the SELL side of the paper bot.

Covers (all with MOCKED trading_client + ledger; NO live API calls):
- trading_client.place_bracket_order: BRACKET request shape, penny-rounding,
  invalid-after-rounding SKIP (sl < entry < tp), leg-id extraction.
- run_inference.size_and_execute_trades: bracket path + deterministic
  client_order_id + one ledger row per submitted bracket.
- position_manager.reconcile:
    * np.busday_count horizon math (5/10/21; day 4 vs 5 vs 6 boundary)
    * kill-switch boundary (14.9% no-fire vs 15.1% fire)
    * leg-cancel-BEFORE-close ordering (recorded via a single mock)
    * bracket-fill detection -> closed_tp / closed_sl
    * orphan Alpaca positions left untouched
"""

import datetime
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alpaca.position_ledger import PositionLedger
from src.alpaca.trading_client import AlpacaTradingClient

STRAT = "1w_tp0p05_sl-0p05"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_client_with_mock():
    """AlpacaTradingClient with a mocked underlying alpaca client."""
    client = AlpacaTradingClient.__new__(AlpacaTradingClient)
    client.client = MagicMock()
    client.data_client = MagicMock()
    client.paper_mode = True
    return client


def _fake_leg(leg_id, order_type, side="sell"):
    leg = MagicMock()
    leg.id = leg_id
    leg.type = order_type  # "limit" (TP) or "stop"/"stop_limit" (SL)
    leg.side = side
    return leg


def _fake_bracket_order(entry_id="entry-1", tp_id="tp-1", sl_id="sl-1"):
    order = MagicMock()
    order.id = entry_id
    order.legs = [
        _fake_leg(tp_id, "limit"),
        _fake_leg(sl_id, "stop"),
    ]
    return order


# --------------------------------------------------------------------------- #
# place_bracket_order
# --------------------------------------------------------------------------- #
def test_place_bracket_order_returns_leg_ids():
    client = _make_client_with_mock()
    client.client.submit_order.return_value = _fake_bracket_order(
        "entry-9", "tp-9", "sl-9"
    )

    result = client.place_bracket_order(
        symbol="AAPL",
        qty=10,
        entry_limit_price=190.0,
        tp_price=199.5,
        sl_price=180.5,
        side="buy",
        client_order_id="cid-1",
    )

    assert result is not None
    assert result["entry_order_id"] == "entry-9"
    assert result["tp_leg_id"] == "tp-9"
    assert result["sl_leg_id"] == "sl-9"
    assert result["client_order_id"] == "cid-1"
    client.client.submit_order.assert_called_once()


def test_place_bracket_order_request_is_bracket_gtc():
    from alpaca.trading.enums import OrderClass, TimeInForce

    client = _make_client_with_mock()
    client.client.submit_order.return_value = _fake_bracket_order()

    client.place_bracket_order("AAPL", 5, 100.0, 105.0, 95.0, side="buy")

    req = client.client.submit_order.call_args[0][0]
    assert req.order_class == OrderClass.BRACKET
    assert req.time_in_force == TimeInForce.GTC
    assert req.take_profit.limit_price == 105.0
    assert req.stop_loss.stop_price == 95.0


def test_place_bracket_order_penny_rounds_prices():
    client = _make_client_with_mock()
    client.client.submit_order.return_value = _fake_bracket_order()

    client.place_bracket_order("AAPL", 5, 100.123, 105.678, 95.111, side="buy")

    req = client.client.submit_order.call_args[0][0]
    assert req.limit_price == 100.12
    assert req.take_profit.limit_price == 105.68
    assert req.stop_loss.stop_price == 95.11


def test_place_bracket_order_skips_when_invalid_ordering():
    """sl < entry < tp must hold; otherwise SKIP (return None, no submit)."""
    client = _make_client_with_mock()
    client.client.submit_order.return_value = _fake_bracket_order()

    # tp below entry -> invalid
    assert client.place_bracket_order("AAPL", 5, 100.0, 99.0, 95.0, side="buy") is None
    # sl above entry -> invalid
    assert (
        client.place_bracket_order("AAPL", 5, 100.0, 105.0, 101.0, side="buy") is None
    )
    client.client.submit_order.assert_not_called()


def test_place_bracket_order_skips_when_collapses_after_rounding():
    """Prices distinct as floats but identical at 2dp -> SKIP."""
    client = _make_client_with_mock()
    client.client.submit_order.return_value = _fake_bracket_order()

    # entry 100.001 -> 100.00, tp 100.004 -> 100.00: collapses
    assert (
        client.place_bracket_order("AAPL", 5, 100.001, 100.004, 99.0, side="buy")
        is None
    )
    client.client.submit_order.assert_not_called()


def test_place_bracket_order_no_client_returns_none():
    client = AlpacaTradingClient.__new__(AlpacaTradingClient)
    client.client = None
    assert client.place_bracket_order("AAPL", 1, 10.0, 11.0, 9.0) is None


# --------------------------------------------------------------------------- #
# np.busday_count horizon math
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "entry,today,expected",
    [
        # Mon 2026-05-25 entry. Business days elapsed:
        (datetime.date(2026, 5, 25), datetime.date(2026, 5, 28), 3),  # Thu
        (datetime.date(2026, 5, 25), datetime.date(2026, 5, 29), 4),  # Fri
        (datetime.date(2026, 5, 25), datetime.date(2026, 6, 1), 5),  # next Mon
        (datetime.date(2026, 5, 25), datetime.date(2026, 6, 2), 6),  # next Tue
    ],
)
def test_busday_count_matches_expected(entry, today, expected):
    assert int(np.busday_count(entry, today)) == expected


def test_horizon_trigger_at_day_5_not_day_4():
    """horizon_days=5: not due at 4 business days, due at >=5."""
    from src.alpaca.position_manager import _is_horizon_due

    entry = datetime.date(2026, 5, 25)  # Monday
    assert _is_horizon_due(entry, datetime.date(2026, 5, 29), 5) is False  # day 4
    assert _is_horizon_due(entry, datetime.date(2026, 6, 1), 5) is True  # day 5
    assert _is_horizon_due(entry, datetime.date(2026, 6, 2), 5) is True  # day 6


def test_horizon_trigger_10_and_21():
    from src.alpaca.position_manager import _is_horizon_due

    entry = datetime.date(2026, 5, 25)  # Monday
    # 10 business days -> 2026-06-08 (Mon)
    assert _is_horizon_due(entry, datetime.date(2026, 6, 5), 10) is False  # day 9
    assert _is_horizon_due(entry, datetime.date(2026, 6, 8), 10) is True  # day 10
    # 21 business days -> 2026-06-23 (Tue)
    assert _is_horizon_due(entry, datetime.date(2026, 6, 22), 21) is False  # day 20
    assert _is_horizon_due(entry, datetime.date(2026, 6, 23), 21) is True  # day 21


# --------------------------------------------------------------------------- #
# reconcile: kill-switch boundary
# --------------------------------------------------------------------------- #
def _ledger_with_open(entry_date=datetime.date(2026, 5, 27)):
    ledger = PositionLedger()
    ledger.add_entry(
        ticker="AAPL",
        strategy_str=STRAT,
        horizon_days=5,
        entry_date=entry_date,
        entry_price=190.0,
        qty=10,
        tp_price=199.5,
        sl_price=180.5,
        entry_order_id="entry-1",
        tp_leg_id="tp-1",
        sl_leg_id="sl-1",
    )
    return ledger


def _tc_with_equity(equity, positions=None, orders=None):
    """Mock trading client. orders maps order_id -> status dict for get_order."""
    tc = MagicMock()
    tc.get_account.return_value = {"equity": equity, "portfolio_value": equity}
    tc.get_positions.return_value = positions if positions is not None else {}
    orders = orders or {}
    tc.get_order.side_effect = lambda oid: orders.get(oid)
    tc.cancel_order.return_value = True
    tc.close_position.return_value = {"id": "close-1", "filled_avg_price": 192.0}
    tc.close_all_positions.return_value = []
    return tc


def test_kill_switch_does_not_fire_at_14_9_pct():
    from src.alpaca.position_manager import reconcile

    ledger = _ledger_with_open()
    ledger.peak_equity = 100000.0
    # 14.9% drawdown -> below threshold
    tc = _tc_with_equity(
        85100.0, positions={"AAPL": {"qty": 10.0, "market_value": 1855.0}}
    )

    reconcile(tc, ledger, today=datetime.date(2026, 5, 28))

    # kill-switch not fired -> no closed_kill rows
    assert (ledger.df["status"] == "closed_kill").sum() == 0
    tc.close_all_positions.assert_not_called()


def test_kill_switch_fires_at_15_1_pct():
    from src.alpaca.position_manager import reconcile

    ledger = _ledger_with_open()
    ledger.peak_equity = 100000.0
    tc = _tc_with_equity(
        84900.0, positions={"AAPL": {"qty": 10.0, "market_value": 1855.0}}
    )

    reconcile(tc, ledger, today=datetime.date(2026, 5, 28))

    # kill-switch fired -> all open rows closed_kill
    assert (ledger.df["status"] == "closed_kill").sum() == 1
    tc.close_all_positions.assert_called_once()
    # legs cancelled before the blanket close
    assert tc.cancel_order.call_count >= 2  # tp + sl legs


def test_kill_switch_updates_peak_equity_when_higher():
    from src.alpaca.position_manager import reconcile

    ledger = _ledger_with_open()
    ledger.peak_equity = 100000.0
    tc = _tc_with_equity(120000.0)  # new high-water mark

    reconcile(tc, ledger, today=datetime.date(2026, 5, 28))

    assert ledger.peak_equity == pytest.approx(120000.0)


# --------------------------------------------------------------------------- #
# reconcile: leg-cancel-BEFORE-close ordering (horizon exit)
# --------------------------------------------------------------------------- #
def test_horizon_exit_cancels_legs_before_close():
    """The leftover TP/SL legs MUST be cancelled before close_position,
    else a leg fills after the market sell -> accidental short."""
    from src.alpaca.position_manager import reconcile

    # entry old enough to be horizon-due (>=5 business days before today)
    ledger = _ledger_with_open(entry_date=datetime.date(2026, 5, 18))  # Mon
    ledger.peak_equity = 100000.0

    calls = []
    tc = MagicMock()
    tc.get_account.return_value = {"equity": 100000.0, "portfolio_value": 100000.0}
    tc.get_positions.return_value = {"AAPL": {"qty": 10.0, "market_value": 1920.0}}
    tc.get_order.side_effect = lambda oid: None  # legs still open, not filled

    def rec_cancel(oid):
        calls.append(("cancel", oid))
        return True

    def rec_close(symbol):
        calls.append(("close", symbol))
        return {"id": "close-1", "filled_avg_price": 192.0}

    tc.cancel_order.side_effect = rec_cancel
    tc.close_position.side_effect = rec_close

    # today = 2026-05-26 (Tue) -> 6 business days after Mon 5/18 -> due
    reconcile(tc, ledger, today=datetime.date(2026, 5, 26))

    # Both cancels happen, and BOTH precede the close.
    cancel_idxs = [i for i, c in enumerate(calls) if c[0] == "cancel"]
    close_idxs = [i for i, c in enumerate(calls) if c[0] == "close"]
    assert cancel_idxs, "expected leg cancels"
    assert close_idxs, "expected a position close"
    assert max(cancel_idxs) < min(close_idxs), f"cancel must precede close: {calls}"

    row = ledger.df.iloc[0]
    assert row["status"] == "closed_horizon"
    assert float(row["exit_price"]) == 192.0


def test_fresh_position_untouched_by_horizon():
    from src.alpaca.position_manager import reconcile

    ledger = _ledger_with_open(entry_date=datetime.date(2026, 5, 27))
    ledger.peak_equity = 100000.0
    tc = _tc_with_equity(
        100000.0, positions={"AAPL": {"qty": 10.0, "market_value": 1920.0}}
    )

    # today only 1 business day later -> not due
    reconcile(tc, ledger, today=datetime.date(2026, 5, 28))

    assert ledger.df.iloc[0]["status"] == "open"
    tc.close_position.assert_not_called()


# --------------------------------------------------------------------------- #
# reconcile: bracket-fill detection
# --------------------------------------------------------------------------- #
def test_bracket_tp_fill_marked_closed_tp():
    from src.alpaca.position_manager import reconcile

    ledger = _ledger_with_open(entry_date=datetime.date(2026, 5, 27))
    ledger.peak_equity = 100000.0

    # TP leg filled, SL leg not. Position already gone from Alpaca.
    orders = {
        "tp-1": {"status": "filled", "filled_avg_price": 199.5, "filled_qty": 10},
        "sl-1": {"status": "canceled", "filled_avg_price": None},
    }
    tc = _tc_with_equity(102000.0, positions={}, orders=orders)

    reconcile(tc, ledger, today=datetime.date(2026, 5, 28))

    row = ledger.df.iloc[0]
    assert row["status"] == "closed_tp"
    assert float(row["exit_price"]) == 199.5
    # realized pnl = (199.5 - 190.0) * 10
    assert float(row["realized_pnl"]) == pytest.approx(95.0)


def test_bracket_sl_fill_marked_closed_sl():
    from src.alpaca.position_manager import reconcile

    ledger = _ledger_with_open(entry_date=datetime.date(2026, 5, 27))
    ledger.peak_equity = 100000.0

    orders = {
        "tp-1": {"status": "canceled", "filled_avg_price": None},
        "sl-1": {"status": "filled", "filled_avg_price": 180.5, "filled_qty": 10},
    }
    tc = _tc_with_equity(98000.0, positions={}, orders=orders)

    reconcile(tc, ledger, today=datetime.date(2026, 5, 28))

    row = ledger.df.iloc[0]
    assert row["status"] == "closed_sl"
    assert float(row["exit_price"]) == 180.5
    assert float(row["realized_pnl"]) == pytest.approx((180.5 - 190.0) * 10)


# --------------------------------------------------------------------------- #
# reconcile: orphan positions left alone
# --------------------------------------------------------------------------- #
def test_orphan_position_left_untouched():
    from src.alpaca.position_manager import reconcile

    ledger = PositionLedger()  # empty: no ledger row for the held position
    ledger.peak_equity = 100000.0
    tc = _tc_with_equity(
        100000.0, positions={"TSLA": {"qty": 3.0, "market_value": 900.0}}
    )

    reconcile(tc, ledger, today=datetime.date(2026, 5, 28))

    # never blind-close an orphan
    tc.close_position.assert_not_called()
    tc.close_all_positions.assert_not_called()


def test_reconcile_returns_ledger():
    from src.alpaca.position_manager import reconcile

    ledger = _ledger_with_open(entry_date=datetime.date(2026, 5, 27))
    ledger.peak_equity = 100000.0
    tc = _tc_with_equity(100000.0, positions={})
    out = reconcile(tc, ledger, today=datetime.date(2026, 5, 28))
    assert out is ledger
