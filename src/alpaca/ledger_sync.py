# src/alpaca/ledger_sync.py
"""
Stage 5: Drive <-> PositionLedger sync helpers shared by run_inference.py
(entry pass) and run_reconcile.py (exit pass).

The ledger parquet lives as a SINGLE Drive file identified by
config.GDRIVE_LEDGER_FILE_ID. Each run downloads it, mutates in memory, and
overwrites the same file id (GoogleDriveClient.upload_or_update). When Drive is
unavailable (no creds / no file id) we fall back to a local parquet under
config.LOG_DIR so the bot still works in local/dev runs — it just won't share
state across ephemeral CI runners.

These helpers are storage-glue only; all ledger semantics live in PositionLedger
and all Drive I/O in GoogleDriveClient.
"""

from pathlib import Path
from typing import Optional

from src import config
from src.alpaca.position_ledger import PositionLedger

# Local fallback path when Drive is not configured.
LOCAL_LEDGER_PATH = config.LOG_DIR / "position_ledger.parquet"


def download_ledger(gdrive_client) -> PositionLedger:
    """Load the ledger from Drive (by file id) or the local fallback.

    Order of precedence:
      1. Drive file id configured + Drive connected -> download into memory.
      2. Otherwise -> local parquet (empty ledger if it doesn't exist).
    """
    file_id = getattr(config, "GDRIVE_LEDGER_FILE_ID", "") or ""

    if file_id and gdrive_client is not None and gdrive_client.is_connected():
        tmp = config.LOG_DIR / "_ledger_download.parquet"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        if gdrive_client.download_file(file_id, tmp):
            try:
                ledger = PositionLedger.load(tmp)
                print(
                    f"[INFO] Loaded ledger from Drive ({len(ledger.df)} rows, "
                    f"peak_equity ${ledger.peak_equity:,.2f})"
                )
                return ledger
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Failed to parse downloaded ledger: {e}")
        else:
            print("[WARN] Ledger download failed; starting from local/empty.")

    ledger = PositionLedger.load(LOCAL_LEDGER_PATH)
    print(f"[INFO] Loaded ledger from local fallback ({len(ledger.df)} rows)")
    return ledger


def upload_ledger(gdrive_client, ledger: PositionLedger) -> Optional[str]:
    """Persist the ledger to Drive (overwrite the same file id) + local copy.

    Always writes the local fallback first so state survives even if Drive is
    down. Returns the Drive file id on success, else None.
    """
    LOCAL_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.save(LOCAL_LEDGER_PATH)

    file_id = getattr(config, "GDRIVE_LEDGER_FILE_ID", "") or ""
    if gdrive_client is None or not gdrive_client.is_connected():
        print("[WARN] Drive not connected; ledger saved locally only.")
        return None

    result_id = gdrive_client.upload_or_update(
        Path(LOCAL_LEDGER_PATH),
        file_id=file_id or None,
        name="position_ledger.parquet",
    )
    if result_id:
        print(f"[INFO] Ledger uploaded to Drive -> {result_id}")
        if not file_id:
            print(
                "[WARN] GDRIVE_LEDGER_FILE_ID was empty; a NEW Drive file was "
                f"created (id={result_id}). Set GDRIVE_LEDGER_FILE_ID={result_id} "
                "so future runs overwrite it in place."
            )
    return result_id
