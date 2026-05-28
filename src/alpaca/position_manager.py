# src/alpaca/position_manager.py
"""
Stage 5c: position-manager reconcile loop — the SELL side of the paper bot.

`reconcile` is run once per session (run_reconcile.py) AFTER the entry pass. It
reconciles the persistent PositionLedger against the live Alpaca account and
applies exits, in this fixed order of precedence:

  1. KILL-SWITCH (portfolio-level, checked FIRST). If the realized drawdown from
     the high-water mark reaches MAX_PORTFOLIO_DD, cancel every open bracket leg,
     flatten the whole book (close_all_positions), and mark every still-open
     ledger row ``closed_kill``. peak_equity is refreshed from current equity on
     every run (it can only ratchet up).

  2. BRACKET FILLS. For each open ledger row, check whether its TP or SL leg has
     already filled on Alpaca (get_order). A filled TP -> ``closed_tp``; a filled
     SL -> ``closed_sl``. Exit price/pnl come from the filled leg. No new orders
     are sent — the bracket already did the work.

  3. HORIZON EXITS. For each still-open row whose business-day age has reached its
     ``horizon_days`` (np.busday_count(entry_date, today) >= horizon_days), CANCEL
     its open TP/SL legs FIRST, THEN market-close the position. Cancelling first
     is critical: a leftover leg that fills AFTER the market sell would re-open the
     position as an accidental SHORT. Mark ``closed_horizon``.

  4. ORPHANS. Alpaca positions with NO matching open ledger row are logged and
     LEFT ALONE — their own brackets manage them. We never blind-close a position
     we don't have a ledger record for.

Equity source for the kill-switch: ``trading_client.get_account()["equity"]``
(Alpaca's mark-to-market account equity), with a fallback to ``portfolio_value``
if equity is absent. This is the same figure logged to the Performance sheet.

The function mutates and returns the ledger (caller persists it to Drive).
"""

import datetime
from typing import Optional

import numpy as np

from src import config


def _is_horizon_due(
    entry_date: datetime.date, today: datetime.date, horizon_days: int
) -> bool:
    """True when >= ``horizon_days`` business days have elapsed since entry.

    Uses numpy business-day counting (Mon-Fri, no holiday calendar): an entry on
    a Monday is "5 business days old" on the following Monday.
    """
    if entry_date is None or today is None:
        return False
    elapsed = int(np.busday_count(entry_date, today))
    return elapsed >= int(horizon_days)


def _leg_filled(order: Optional[dict]) -> bool:
    """True when a get_order() result represents a filled leg."""
    if not order:
        return False
    status = str(order.get("status", "")).lower()
    # Alpaca status strings: "filled" / "OrderStatus.FILLED" both contain
    # "filled"; guard against partially_filled by requiring an exact-ish match.
    return status.endswith("filled") and "partially" not in status


def _exit_price_of(order: Optional[dict], fallback: float) -> float:
    """Best-effort exit fill price from a leg order, else the fallback."""
    if order:
        price = order.get("filled_avg_price")
        if price:
            return float(price)
    return float(fallback)


def _equity_of(account: Optional[dict]) -> Optional[float]:
    """Pull mark-to-market equity from an account dict, else portfolio_value."""
    if not account:
        return None
    equity = account.get("equity")
    if equity is None:
        equity = account.get("portfolio_value")
    return float(equity) if equity is not None else None


def reconcile(trading_client, ledger, today: datetime.date = None):
    """Reconcile the ledger against live Alpaca state and apply exits.

    Args:
        trading_client: AlpacaTradingClient (or mock) exposing get_account,
            get_positions, get_order, cancel_order, close_position,
            close_all_positions.
        ledger: PositionLedger to mutate in place.
        today: Override "today" (defaults to the local date). Used by tests.

    Returns:
        The same ledger instance (mutated).
    """
    if today is None:
        today = datetime.date.today()

    account = trading_client.get_account()
    equity = _equity_of(account)

    # ---------------------------------------------------------------- #
    # (a) KILL-SWITCH — evaluated FIRST.
    # ---------------------------------------------------------------- #
    if equity is not None:
        peak = max(float(ledger.peak_equity or 0.0), equity)
        drawdown = (peak - equity) / peak if peak > 0 else 0.0

        if peak > 0 and drawdown >= config.MAX_PORTFOLIO_DD:
            print(
                f"[KILL-SWITCH] Drawdown {drawdown:.2%} >= "
                f"{config.MAX_PORTFOLIO_DD:.2%} (peak ${peak:,.2f}, "
                f"equity ${equity:,.2f}). Flattening book."
            )
            open_rows = ledger.open_positions()

            # Cancel every open bracket leg first so no leg can re-open a
            # position after the blanket close.
            for _, row in open_rows.iterrows():
                for leg_col in ("tp_leg_id", "sl_leg_id"):
                    leg_id = row.get(leg_col)
                    if leg_id is not None and not _is_na(leg_id):
                        trading_client.cancel_order(str(leg_id))

            trading_client.close_all_positions()

            # Resolve a per-row exit price from live positions where possible.
            positions = trading_client.get_positions() or {}
            for _, row in open_rows.iterrows():
                ticker = row["ticker"]
                pos = positions.get(ticker) or {}
                exit_price = float(
                    pos.get("current_price")
                    or pos.get("avg_entry_price")
                    or row["entry_price"]
                )
                pnl = (exit_price - float(row["entry_price"])) * int(row["qty"])
                ledger.mark_closed(
                    row["client_order_id"],
                    status="closed_kill",
                    exit_date=today,
                    exit_price=exit_price,
                    realized_pnl=pnl,
                )

            ledger.peak_equity = equity  # reset HWM to current after de-risk
            return ledger

        # No kill: ratchet the high-water mark up.
        ledger.peak_equity = peak

    # ---------------------------------------------------------------- #
    # (b) BRACKET FILLS + (c) HORIZON EXITS, per open ledger row.
    # ---------------------------------------------------------------- #
    positions = trading_client.get_positions() or {}
    ledger_tickers = set()

    open_rows = ledger.open_positions()
    for _, row in open_rows.iterrows():
        cid = row["client_order_id"]
        ticker = row["ticker"]
        ledger_tickers.add(ticker)
        entry_price = float(row["entry_price"])
        qty = int(row["qty"])

        tp_leg_id = row.get("tp_leg_id")
        sl_leg_id = row.get("sl_leg_id")

        # (b) Detect a bracket leg that already filled.
        tp_order = (
            trading_client.get_order(str(tp_leg_id))
            if tp_leg_id is not None and not _is_na(tp_leg_id)
            else None
        )
        sl_order = (
            trading_client.get_order(str(sl_leg_id))
            if sl_leg_id is not None and not _is_na(sl_leg_id)
            else None
        )

        if _leg_filled(tp_order):
            exit_price = _exit_price_of(tp_order, float(row["tp_price"]))
            ledger.mark_closed(
                cid,
                status="closed_tp",
                exit_date=today,
                exit_price=exit_price,
                realized_pnl=(exit_price - entry_price) * qty,
            )
            continue

        if _leg_filled(sl_order):
            exit_price = _exit_price_of(sl_order, float(row["sl_price"]))
            ledger.mark_closed(
                cid,
                status="closed_sl",
                exit_date=today,
                exit_price=exit_price,
                realized_pnl=(exit_price - entry_price) * qty,
            )
            continue

        # (c) Horizon exit.
        if _is_horizon_due(row["entry_date"], today, int(row["horizon_days"])):
            # Cancel the still-open legs BEFORE closing, else a leg could fill
            # after the market sell and re-open the position as a short.
            for leg_id in (tp_leg_id, sl_leg_id):
                if leg_id is not None and not _is_na(leg_id):
                    trading_client.cancel_order(str(leg_id))

            close_result = trading_client.close_position(ticker)
            pos = positions.get(ticker) or {}
            exit_price = float(
                (close_result or {}).get("filled_avg_price")
                or pos.get("current_price")
                or pos.get("avg_entry_price")
                or entry_price
            )
            ledger.mark_closed(
                cid,
                status="closed_horizon",
                exit_date=today,
                exit_price=exit_price,
                realized_pnl=(exit_price - entry_price) * qty,
            )

    # ---------------------------------------------------------------- #
    # (d) ORPHANS — log and leave to their own brackets.
    # ---------------------------------------------------------------- #
    open_after = set(ledger.open_positions()["ticker"])
    for ticker in positions:
        if ticker not in ledger_tickers and ticker not in open_after:
            print(
                f"[ORPHAN] Alpaca position {ticker} has no open ledger row; "
                f"leaving it to its own bracket (never blind-closing)."
            )

    return ledger


def _is_na(value) -> bool:
    """True for None/NaN/NA scalars (leg ids may be missing on a ledger row)."""
    if value is None:
        return True
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
