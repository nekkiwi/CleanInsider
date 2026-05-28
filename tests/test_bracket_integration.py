# tests/test_bracket_integration.py
"""
Stage 5 exits: LIVE paper-mode bracket integration test.

SELF-SKIPS when ALPACA_API_KEY / ALPACA_SECRET_KEY are absent (so CI without
credentials does not place orders). Also marked `integration` so the CI suite
(`-m 'not integration'`) deselects it regardless.

Places a real 1-share AAPL bracket on the paper account, asserts the TP + SL
legs exist, then cancels the legs and closes the position, asserting flat.
"""

import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alpaca.trading_client import AlpacaTradingClient

pytestmark = pytest.mark.integration

_HAS_KEYS = bool(os.environ.get("ALPACA_API_KEY")) and bool(
    os.environ.get("ALPACA_SECRET_KEY")
)


@pytest.mark.skipif(not _HAS_KEYS, reason="no Alpaca credentials in env")
def test_paper_bracket_roundtrip():
    client = AlpacaTradingClient()
    if not client.is_connected():
        pytest.skip("Alpaca client not connected")
    if not client.paper_mode:
        pytest.skip("refusing to place bracket outside paper mode")

    symbol = "AAPL"
    prices = client.get_latest_prices([symbol])
    mid = prices.get(symbol)
    if not mid:
        pytest.skip("no live quote for AAPL")

    # Entry far below market so it rests (won't fill); TP above, SL below entry.
    entry = round(mid * 0.5, 2)
    tp = round(entry * 1.05, 2)
    sl = round(entry * 0.95, 2)

    result = client.place_bracket_order(
        symbol=symbol,
        qty=1,
        entry_limit_price=entry,
        tp_price=tp,
        sl_price=sl,
        side="buy",
    )

    assert result is not None, "bracket order submission failed"
    entry_id = result["entry_order_id"]
    assert entry_id

    try:
        time.sleep(1)
        parent = client.get_order(entry_id)
        assert parent is not None
        # The TP/SL leg ids were captured at submit time.
        assert result["tp_leg_id"] is not None
        assert result["sl_leg_id"] is not None
    finally:
        # Cancel legs first, then the parent / close, then assert flat.
        for leg_key in ("tp_leg_id", "sl_leg_id"):
            leg_id = result.get(leg_key)
            if leg_id:
                client.cancel_order(leg_id)
        client.cancel_order(entry_id)
        time.sleep(1)
        # If anything filled, flatten.
        if client.get_position(symbol):
            client.close_position(symbol)
            time.sleep(1)

        assert client.get_position(symbol) is None
