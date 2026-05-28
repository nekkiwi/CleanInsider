# tests/test_ledger.py
"""
Stage 5a: Drive-backed position ledger + Google Drive update-by-id.

Covers:
- PositionLedger parquet round-trip (all columns + dtypes), peak_equity persists
- UPSERT idempotency on client_order_id (no duplicate rows)
- status transitions: open -> mark_closed; open_positions() filtering
- deterministic client_order_id format for (strategy, ticker, entry_date)
- GoogleDriveClient.upload_or_update: files().update when file_id given,
  files().create when not (mocked Drive service, no network)
"""

import datetime
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alpaca.position_ledger import PositionLedger


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _entry_kwargs(**overrides):
    """Default add_entry kwargs; override individual fields per-test."""
    base = dict(
        ticker="AAPL",
        strategy_str="1w_tp0p05_sl-0p05",
        horizon_days=5,
        entry_date=datetime.date(2026, 5, 27),
        entry_price=190.0,
        qty=10,
        tp_price=199.5,
        sl_price=180.5,
        entry_order_id="ord-entry-1",
        tp_leg_id="ord-tp-1",
        sl_leg_id="ord-sl-1",
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# client_order_id determinism
# --------------------------------------------------------------------------- #
def test_client_order_id_is_deterministic():
    cid1 = PositionLedger.make_client_order_id(
        "1w_tp0p05_sl-0p05", "AAPL", datetime.date(2026, 5, 27)
    )
    cid2 = PositionLedger.make_client_order_id(
        "1w_tp0p05_sl-0p05", "AAPL", datetime.date(2026, 5, 27)
    )
    assert cid1 == cid2
    assert cid1 == "1w_tp0p05_sl-0p05|AAPL|2026-05-27"


def test_client_order_id_differs_by_field():
    base = PositionLedger.make_client_order_id(
        "1w_tp0p05_sl-0p05", "AAPL", datetime.date(2026, 5, 27)
    )
    assert base != PositionLedger.make_client_order_id(
        "2w_tp0p05_sl-0p05", "AAPL", datetime.date(2026, 5, 27)
    )
    assert base != PositionLedger.make_client_order_id(
        "1w_tp0p05_sl-0p05", "MSFT", datetime.date(2026, 5, 27)
    )
    assert base != PositionLedger.make_client_order_id(
        "1w_tp0p05_sl-0p05", "AAPL", datetime.date(2026, 5, 28)
    )


def test_add_entry_returns_deterministic_client_order_id():
    ledger = PositionLedger()
    cid = ledger.add_entry(**_entry_kwargs())
    assert cid == "1w_tp0p05_sl-0p05|AAPL|2026-05-27"


# --------------------------------------------------------------------------- #
# round-trip + dtypes + peak_equity persistence
# --------------------------------------------------------------------------- #
def test_round_trip_all_columns_and_dtypes(tmp_path):
    path = tmp_path / "ledger.parquet"
    ledger = PositionLedger()
    cid = ledger.add_entry(**_entry_kwargs())
    ledger.peak_equity = 123456.78
    ledger.save(path)

    reloaded = PositionLedger.load(path)

    assert len(reloaded.df) == 1
    row = reloaded.df.iloc[0]
    assert row["client_order_id"] == cid
    assert row["ticker"] == "AAPL"
    assert row["strategy_str"] == "1w_tp0p05_sl-0p05"
    assert int(row["horizon_days"]) == 5
    assert row["entry_date"] == datetime.date(2026, 5, 27)
    assert float(row["entry_price"]) == 190.0
    assert int(row["qty"]) == 10
    assert float(row["tp_price"]) == 199.5
    assert float(row["sl_price"]) == 180.5
    assert row["entry_order_id"] == "ord-entry-1"
    assert row["tp_leg_id"] == "ord-tp-1"
    assert row["sl_leg_id"] == "ord-sl-1"
    assert row["status"] == "open"
    # peak_equity persists across save/load
    assert reloaded.peak_equity == pytest.approx(123456.78)

    # All required columns present
    expected_cols = set(PositionLedger.COLUMNS)
    assert expected_cols.issubset(set(reloaded.df.columns))

    # dtypes: horizon_days/qty integer, prices float
    assert pd.api.types.is_integer_dtype(reloaded.df["horizon_days"])
    assert pd.api.types.is_integer_dtype(reloaded.df["qty"])
    assert pd.api.types.is_float_dtype(reloaded.df["entry_price"])


def test_peak_equity_default_zero_and_load_empty(tmp_path):
    ledger = PositionLedger()
    assert ledger.peak_equity == 0.0
    path = tmp_path / "empty.parquet"
    ledger.save(path)
    reloaded = PositionLedger.load(path)
    assert reloaded.peak_equity == 0.0
    assert len(reloaded.df) == 0
    # Loading a non-existent path yields an empty ledger
    missing = PositionLedger.load(tmp_path / "does_not_exist.parquet")
    assert len(missing.df) == 0
    assert missing.peak_equity == 0.0


# --------------------------------------------------------------------------- #
# UPSERT idempotency
# --------------------------------------------------------------------------- #
def test_add_entry_upsert_idempotent(tmp_path):
    ledger = PositionLedger()
    ledger.add_entry(**_entry_kwargs(qty=10, entry_price=190.0))
    # Re-add same key (same strategy/ticker/entry_date) with new values
    ledger.add_entry(**_entry_kwargs(qty=20, entry_price=191.5, entry_order_id="ord-2"))

    assert len(ledger.df) == 1
    row = ledger.df.iloc[0]
    assert int(row["qty"]) == 20
    assert float(row["entry_price"]) == 191.5
    assert row["entry_order_id"] == "ord-2"

    # Round-trip still single row
    path = tmp_path / "ledger.parquet"
    ledger.save(path)
    reloaded = PositionLedger.load(path)
    assert len(reloaded.df) == 1


def test_distinct_keys_create_distinct_rows():
    ledger = PositionLedger()
    ledger.add_entry(**_entry_kwargs(ticker="AAPL"))
    ledger.add_entry(**_entry_kwargs(ticker="MSFT"))
    assert len(ledger.df) == 2


# --------------------------------------------------------------------------- #
# status transitions / open_positions
# --------------------------------------------------------------------------- #
def test_open_positions_and_mark_closed():
    ledger = PositionLedger()
    cid_a = ledger.add_entry(**_entry_kwargs(ticker="AAPL"))
    ledger.add_entry(**_entry_kwargs(ticker="MSFT"))

    open_before = ledger.open_positions()
    assert len(open_before) == 2
    assert set(open_before["status"]) == {"open"}

    ledger.mark_closed(
        cid_a,
        status="closed_tp",
        exit_date=datetime.date(2026, 6, 1),
        exit_price=199.5,
        realized_pnl=95.0,
    )

    closed_row = ledger.df[ledger.df["client_order_id"] == cid_a].iloc[0]
    assert closed_row["status"] == "closed_tp"
    assert closed_row["exit_date"] == datetime.date(2026, 6, 1)
    assert float(closed_row["exit_price"]) == 199.5
    assert float(closed_row["realized_pnl"]) == 95.0

    open_after = ledger.open_positions()
    assert len(open_after) == 1
    assert open_after.iloc[0]["ticker"] == "MSFT"


def test_mark_closed_unknown_key_raises():
    ledger = PositionLedger()
    ledger.add_entry(**_entry_kwargs())
    with pytest.raises(KeyError):
        ledger.mark_closed(
            "nonexistent|key|2026-01-01",
            status="closed_horizon",
            exit_date=datetime.date(2026, 6, 1),
            exit_price=1.0,
            realized_pnl=0.0,
        )


def test_mark_closed_status_round_trips(tmp_path):
    ledger = PositionLedger()
    cid = ledger.add_entry(**_entry_kwargs())
    ledger.mark_closed(
        cid,
        status="closed_horizon",
        exit_date=datetime.date(2026, 6, 3),
        exit_price=192.0,
        realized_pnl=20.0,
    )
    path = tmp_path / "ledger.parquet"
    ledger.save(path)
    reloaded = PositionLedger.load(path)
    row = reloaded.df.iloc[0]
    assert row["status"] == "closed_horizon"
    assert row["exit_date"] == datetime.date(2026, 6, 3)
    assert float(row["realized_pnl"]) == 20.0
    assert len(reloaded.open_positions()) == 0


def test_invalid_status_rejected():
    ledger = PositionLedger()
    cid = ledger.add_entry(**_entry_kwargs())
    with pytest.raises(ValueError):
        ledger.mark_closed(
            cid,
            status="not_a_valid_status",
            exit_date=datetime.date(2026, 6, 3),
            exit_price=192.0,
            realized_pnl=20.0,
        )


# --------------------------------------------------------------------------- #
# bytes round-trip (storage-agnostic)
# --------------------------------------------------------------------------- #
def test_bytes_round_trip():
    ledger = PositionLedger()
    ledger.add_entry(**_entry_kwargs())
    ledger.peak_equity = 50000.0
    raw = ledger.to_bytes()
    assert isinstance(raw, (bytes, bytearray))

    reloaded = PositionLedger.from_bytes(raw)
    assert len(reloaded.df) == 1
    assert reloaded.df.iloc[0]["ticker"] == "AAPL"
    assert reloaded.peak_equity == pytest.approx(50000.0)


# --------------------------------------------------------------------------- #
# GoogleDriveClient.upload_or_update (mocked Drive service, no network)
# --------------------------------------------------------------------------- #
def _make_gdrive_with_mock_service():
    from src.alpaca.google_drive import GoogleDriveClient

    client = GoogleDriveClient.__new__(GoogleDriveClient)
    client.drive_service = MagicMock()
    client.models_folder_id = "models-folder"
    client.log_sheet_id = ""
    client.sheets_service = None
    return client


def test_upload_or_update_calls_update_when_file_id(tmp_path):
    client = _make_gdrive_with_mock_service()
    files = client.drive_service.files.return_value
    files.update.return_value.execute.return_value = {"id": "existing-id"}

    local = tmp_path / "ledger.parquet"
    local.write_bytes(b"parquet-bytes")

    result = client.upload_or_update(local, file_id="existing-id")

    assert result == "existing-id"
    # update called with the right fileId; create NOT called
    files.update.assert_called_once()
    _, kwargs = files.update.call_args
    assert kwargs["fileId"] == "existing-id"
    assert "media_body" in kwargs
    files.create.assert_not_called()


def test_upload_or_update_calls_create_when_no_file_id(tmp_path):
    client = _make_gdrive_with_mock_service()
    files = client.drive_service.files.return_value
    files.create.return_value.execute.return_value = {"id": "new-id"}

    local = tmp_path / "ledger.parquet"
    local.write_bytes(b"parquet-bytes")

    result = client.upload_or_update(
        local, file_id=None, name="ledger.parquet", folder_id="models-folder"
    )

    assert result == "new-id"
    files.create.assert_called_once()
    _, kwargs = files.create.call_args
    assert kwargs["body"]["name"] == "ledger.parquet"
    assert kwargs["body"]["parents"] == ["models-folder"]
    files.update.assert_not_called()


def test_upload_or_update_no_service_returns_none(tmp_path):
    from src.alpaca.google_drive import GoogleDriveClient

    client = GoogleDriveClient.__new__(GoogleDriveClient)
    client.drive_service = None
    client.models_folder_id = "models-folder"

    local = tmp_path / "ledger.parquet"
    local.write_bytes(b"x")
    assert client.upload_or_update(local, file_id="x") is None
