# tests/test_reconcile_e2e.py
"""
Stage 5 exits: end-to-end reconcile dry-run on a synthetic two-row ledger.

NO live API: trading_client is fully mocked (records cancel/close call order),
Google Drive is not touched. Asserts:
  * the horizon-EXPIRED row is closed (legs cancelled THEN market-closed)
  * the FRESH row is left untouched (still open)
  * the persisted ledger round-trips with the new statuses
"""

import datetime
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alpaca.position_ledger import PositionLedger
from src.alpaca.position_manager import reconcile

STRAT = "1w_tp0p05_sl-0p05"


def _two_row_ledger():
    ledger = PositionLedger()
    ledger.peak_equity = 100000.0
    # EXPIRED: Mon 2026-05-18, horizon 5bd -> due well before 2026-05-28.
    ledger.add_entry(
        ticker="OLD",
        strategy_str=STRAT,
        horizon_days=5,
        entry_date=datetime.date(2026, 5, 18),
        entry_price=50.0,
        qty=20,
        tp_price=52.5,
        sl_price=47.5,
        entry_order_id="old-entry",
        tp_leg_id="old-tp",
        sl_leg_id="old-sl",
    )
    # FRESH: 2026-05-27, only ~1 business day before today.
    ledger.add_entry(
        ticker="NEW",
        strategy_str=STRAT,
        horizon_days=5,
        entry_date=datetime.date(2026, 5, 27),
        entry_price=100.0,
        qty=5,
        tp_price=105.0,
        sl_price=95.0,
        entry_order_id="new-entry",
        tp_leg_id="new-tp",
        sl_leg_id="new-sl",
    )
    return ledger


def test_reconcile_dry_run_expires_old_keeps_new(tmp_path):
    ledger = _two_row_ledger()

    calls = []
    tc = MagicMock()
    tc.get_account.return_value = {"equity": 100000.0, "portfolio_value": 100000.0}
    # Both positions still held on Alpaca.
    tc.get_positions.return_value = {
        "OLD": {"qty": 20.0, "market_value": 1040.0},
        "NEW": {"qty": 5.0, "market_value": 510.0},
    }
    tc.get_order.side_effect = lambda oid: None  # no bracket leg has filled

    def rec_cancel(oid):
        calls.append(("cancel", oid))
        return True

    def rec_close(symbol):
        calls.append(("close", symbol))
        return {"id": f"close-{symbol}", "filled_avg_price": 52.0}

    tc.cancel_order.side_effect = rec_cancel
    tc.close_position.side_effect = rec_close

    out = reconcile(tc, ledger, today=datetime.date(2026, 5, 28))

    # OLD closed via horizon, NEW still open.
    old = out.df[out.df["ticker"] == "OLD"].iloc[0]
    new = out.df[out.df["ticker"] == "NEW"].iloc[0]
    assert old["status"] == "closed_horizon"
    assert new["status"] == "open"

    # Only the OLD position was closed.
    closed_syms = [c[1] for c in calls if c[0] == "close"]
    assert closed_syms == ["OLD"]

    # OLD legs cancelled BEFORE the OLD close.
    old_cancel_idxs = [
        i
        for i, c in enumerate(calls)
        if c[0] == "cancel" and c[1] in ("old-tp", "old-sl")
    ]
    old_close_idx = next(i for i, c in enumerate(calls) if c == ("close", "OLD"))
    assert old_cancel_idxs, "expected OLD leg cancels"
    assert max(old_cancel_idxs) < old_close_idx

    # NEW legs never cancelled.
    assert ("cancel", "new-tp") not in calls
    assert ("cancel", "new-sl") not in calls

    # Round-trips with new statuses.
    path = tmp_path / "ledger.parquet"
    out.save(path)
    reloaded = PositionLedger.load(path)
    statuses = dict(zip(reloaded.df["ticker"], reloaded.df["status"]))
    assert statuses == {"OLD": "closed_horizon", "NEW": "open"}
    assert len(reloaded.open_positions()) == 1
