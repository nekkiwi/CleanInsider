# run_reconcile.py
"""
Stage 5c entry point: the EXIT-side daily run (mirrors run_inference.py).

Flow:
  1. Download the persistent position ledger from Google Drive (local fallback).
  2. reconcile(trading_client, ledger): kill-switch -> bracket fills -> horizon
     exits -> orphan logging. Cancels stale TP/SL legs BEFORE market-closing a
     horizon-expired position (so no leftover leg re-opens it as a short).
  3. Upload the mutated ledger back to the SAME Drive file.
  4. Log a performance row (equity / open positions / peak / drawdown).

Run this AFTER run_inference.py each session.

Usage:
    python run_reconcile.py                 # full reconcile + persist + log
    python run_reconcile.py --dry-run       # reconcile in memory, DO NOT persist
                                            # or place/cancel exit orders
    python run_reconcile.py --model model_1w_tp5_sl5   # log under this model id
"""

import argparse
import datetime
import sys

from src import config
from src.alpaca.google_drive import GoogleDriveClient
from src.alpaca.ledger_sync import download_ledger, upload_ledger
from src.alpaca.position_manager import reconcile
from src.alpaca.trading_client import AlpacaTradingClient


def main():
    parser = argparse.ArgumentParser(
        description="Reconcile the position ledger against Alpaca and apply exits"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Reconcile in memory only: no exit orders, no ledger upload",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="1w_tp0p05_sl-0p05",
        help="Model/strategy id used for performance logging",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CleanInsider Daily Reconcile (exit pass)")
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Paper' if config.PAPER_MODE else 'Live'} Trading")
    if args.dry_run:
        print("Mode: DRY-RUN (no orders, no upload)")
    print("=" * 60)

    gdrive_client = GoogleDriveClient()
    trading_client = AlpacaTradingClient()

    if not trading_client.is_connected():
        print(
            "[ERROR] Alpaca client not connected (missing/invalid ALPACA_API_KEY "
            "/ ALPACA_SECRET_KEY). Cannot reconcile exits. Exiting."
        )
        sys.exit(1)

    # 1. Download ledger.
    ledger = download_ledger(gdrive_client)
    open_before = len(ledger.open_positions())
    print(f"[INFO] {open_before} open position(s) before reconcile")

    if open_before == 0:
        print("[INFO] No open positions to reconcile.")

    if args.dry_run:
        # Reconcile against a deep copy so the real ledger / orders are untouched.
        import copy

        preview = copy.deepcopy(ledger)

        # Neutralize side-effecting methods in dry-run.
        class _NoOpClient:
            def __init__(self, inner):
                self._inner = inner

            def get_account(self):
                return self._inner.get_account()

            def get_positions(self):
                return self._inner.get_positions()

            def get_order(self, oid):
                return self._inner.get_order(oid)

            def cancel_order(self, oid):
                print(f"  [DRY-RUN] would cancel leg {oid}")
                return True

            def close_position(self, symbol):
                print(f"  [DRY-RUN] would close position {symbol}")
                return None

            def close_all_positions(self):
                print("  [DRY-RUN] would close ALL positions (kill-switch)")
                return []

        reconcile(_NoOpClient(trading_client), preview)
        closed = preview.df[preview.df["status"] != "open"]
        print(
            f"[DRY-RUN] reconcile would close {len(closed)} position(s); "
            "ledger NOT persisted."
        )
        sys.exit(0)

    # 2. Reconcile (mutates ledger, may cancel/close real orders).
    reconcile(trading_client, ledger)
    open_after = len(ledger.open_positions())
    print(f"[INFO] {open_after} open position(s) after reconcile")

    # 3. Persist ledger back to Drive.
    upload_ledger(gdrive_client, ledger)

    # 4. Log a performance snapshot.
    account = trading_client.get_account() or {}
    equity = account.get("equity", account.get("portfolio_value", 0))
    peak = ledger.peak_equity or 0.0
    drawdown = (peak - equity) / peak if peak > 0 else 0.0

    if gdrive_client.is_connected():
        try:
            gdrive_client.log_performance(
                {
                    "portfolio_value": account.get("portfolio_value", 0),
                    "equity": equity,
                    "cash": account.get("cash", 0),
                    "num_positions": open_after,
                    "notes": f"reconcile; peak ${peak:,.0f}; dd {drawdown:.2%}",
                },
                model_id=args.model,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Failed to log reconcile performance: {e}")

    print("\n" + "=" * 60)
    print("Reconcile Complete")
    print(f"  Open before/after: {open_before} -> {open_after}")
    print(f"  Peak equity: ${peak:,.2f}  |  Drawdown: {drawdown:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
