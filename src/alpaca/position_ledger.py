# src/alpaca/position_ledger.py
"""
Stage 5a: a persistent position ledger backed by a single parquet file.

The live trading system (run_inference.py) runs on ephemeral GitHub Actions
runners, so any state that must survive between daily runs cannot live on disk.
This ledger is the backbone for time-based exits (Stage 5b/5c): it records every
entry we open, the bracket leg order ids, and the eventual exit, so a fresh
runner can reconstruct what is currently held and what should be exited.

Storage model
-------------
The ledger is a single pandas DataFrame persisted as one parquet file. The
GitHub Actions flow downloads the parquet from Google Drive at the start of a
run, mutates it in memory, and overwrites the same Drive file at the end
(GoogleDriveClient.upload_or_update). This class is deliberately
*storage-agnostic*: it only knows local parquet paths / bytes. Drive sync is the
GoogleDriveClient's job.

Primary key
-----------
client_order_id = f"{strategy_str}|{ticker}|{entry_date:%Y-%m-%d}"

It is deterministic, so an idempotent re-run of the same trading day UPSERTS the
existing row instead of creating a duplicate. It doubles as the Alpaca
client_order_id when the bracket entry is submitted (Stage 5b).

peak_equity
-----------
A single scalar (high-water mark of portfolio equity, used for drawdown-based
de-risking later) is persisted in the parquet's file-level (schema) key-value
metadata under the key ``b"peak_equity"`` rather than as a data row. This keeps
the data rows clean and the column dtypes intact across round-trips.
"""

import datetime
import io
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# parquet schema-metadata key for the persisted peak_equity scalar
_PEAK_EQUITY_META_KEY = b"cleaninsider.peak_equity"

PathLike = Union[str, Path]


class PositionLedger:
    """A single-parquet position ledger with UPSERT semantics.

    The DataFrame is exposed as ``self.df``; callers mutate via the methods
    (add_entry / mark_closed) rather than touching the frame directly.
    """

    # Column order is the canonical schema written to parquet.
    COLUMNS = [
        "client_order_id",
        "entry_order_id",
        "tp_leg_id",
        "sl_leg_id",
        "ticker",
        "strategy_str",
        "horizon_days",
        "entry_date",
        "entry_price",
        "qty",
        "tp_price",
        "sl_price",
        "status",
        "exit_date",
        "exit_price",
        "realized_pnl",
    ]

    # Allowed status values.
    VALID_STATUSES = (
        "open",
        "closed_tp",
        "closed_sl",
        "closed_horizon",
        "closed_kill",
    )

    # Object/string-typed columns (everything not numeric/date).
    _STR_COLUMNS = (
        "client_order_id",
        "entry_order_id",
        "tp_leg_id",
        "sl_leg_id",
        "ticker",
        "strategy_str",
        "status",
    )
    _INT_COLUMNS = ("horizon_days", "qty")
    _FLOAT_COLUMNS = (
        "entry_price",
        "tp_price",
        "sl_price",
        "exit_price",
        "realized_pnl",
    )
    _DATE_COLUMNS = ("entry_date", "exit_date")

    def __init__(
        self, df: Optional[pd.DataFrame] = None, peak_equity: float = 0.0
    ) -> None:
        if df is None:
            df = self._empty_frame()
        self.df = self._coerce_dtypes(df.copy())
        self.peak_equity = float(peak_equity)

    # ------------------------------------------------------------------ #
    # key construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def make_client_order_id(
        strategy_str: str, ticker: str, entry_date: datetime.date
    ) -> str:
        """Deterministic primary key: ``strategy|ticker|YYYY-MM-DD``."""
        if isinstance(entry_date, datetime.datetime):
            entry_date = entry_date.date()
        return f"{strategy_str}|{ticker}|{entry_date:%Y-%m-%d}"

    # ------------------------------------------------------------------ #
    # schema helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _empty_frame(cls) -> pd.DataFrame:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cls.COLUMNS})

    @classmethod
    def _coerce_dtypes(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all canonical columns exist with the expected dtypes."""
        for col in cls.COLUMNS:
            if col not in df.columns:
                df[col] = pd.Series([pd.NA] * len(df), dtype="object")

        # Reorder to the canonical column order.
        df = df[cls.COLUMNS].copy()

        for col in cls._STR_COLUMNS:
            df[col] = df[col].astype("object")

        # Integer columns: nullable Int64 so a fully-empty frame stays integer.
        for col in cls._INT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        for col in cls._FLOAT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        for col in cls._DATE_COLUMNS:
            df[col] = df[col].map(cls._to_date)

        return df

    @staticmethod
    def _to_date(value) -> Optional[datetime.date]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, datetime.datetime):
            return value.date()
        if isinstance(value, datetime.date):
            return value
        if isinstance(value, pd.Timestamp):
            return value.date()
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return pd.Timestamp(value).date()

    # ------------------------------------------------------------------ #
    # mutation
    # ------------------------------------------------------------------ #
    def add_entry(
        self,
        ticker: str,
        strategy_str: str,
        horizon_days: int,
        entry_date: datetime.date,
        entry_price: float,
        qty: int,
        tp_price: float,
        sl_price: float,
        entry_order_id: Optional[str] = None,
        tp_leg_id: Optional[str] = None,
        sl_leg_id: Optional[str] = None,
    ) -> str:
        """Insert or update a position keyed by client_order_id (UPSERT).

        Re-adding the same (strategy_str, ticker, entry_date) updates the
        existing row in place; it never creates a duplicate. New entries start
        with ``status="open"`` and null exit fields. Returns the client_order_id.
        """
        if isinstance(entry_date, datetime.datetime):
            entry_date = entry_date.date()

        client_order_id = self.make_client_order_id(strategy_str, ticker, entry_date)

        row = {
            "client_order_id": client_order_id,
            "entry_order_id": entry_order_id,
            "tp_leg_id": tp_leg_id,
            "sl_leg_id": sl_leg_id,
            "ticker": ticker,
            "strategy_str": strategy_str,
            "horizon_days": int(horizon_days),
            "entry_date": entry_date,
            "entry_price": float(entry_price),
            "qty": int(qty),
            "tp_price": float(tp_price),
            "sl_price": float(sl_price),
            "status": "open",
            "exit_date": None,
            "exit_price": None,
            "realized_pnl": None,
        }

        mask = self.df["client_order_id"] == client_order_id
        if mask.any():
            idx = self.df.index[mask][0]
            for col, val in row.items():
                self.df.at[idx, col] = val
        else:
            new_row = self._coerce_dtypes(pd.DataFrame([row]))
            self.df = pd.concat([self.df, new_row], ignore_index=True)

        self.df = self._coerce_dtypes(self.df)
        return client_order_id

    def mark_closed(
        self,
        client_order_id: str,
        status: str,
        exit_date: datetime.date,
        exit_price: float,
        realized_pnl: float,
    ) -> None:
        """Mark a position closed, setting status + exit fields.

        Raises KeyError if the client_order_id is unknown, ValueError if
        ``status`` is not a recognised closed_* status.
        """
        if status not in self.VALID_STATUSES or status == "open":
            raise ValueError(
                f"Invalid close status {status!r}; expected one of "
                f"{[s for s in self.VALID_STATUSES if s != 'open']}"
            )

        if isinstance(exit_date, datetime.datetime):
            exit_date = exit_date.date()

        mask = self.df["client_order_id"] == client_order_id
        if not mask.any():
            raise KeyError(f"Unknown client_order_id: {client_order_id!r}")

        idx = self.df.index[mask][0]
        self.df.at[idx, "status"] = status
        self.df.at[idx, "exit_date"] = exit_date
        self.df.at[idx, "exit_price"] = float(exit_price)
        self.df.at[idx, "realized_pnl"] = float(realized_pnl)

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def open_positions(self) -> pd.DataFrame:
        """Return a copy of the rows whose status == 'open'."""
        return self.df[self.df["status"] == "open"].copy()

    # ------------------------------------------------------------------ #
    # serialization (storage-agnostic: local path or bytes)
    # ------------------------------------------------------------------ #
    def _to_arrow_table(self) -> pa.Table:
        table = pa.Table.from_pandas(self._coerce_dtypes(self.df), preserve_index=False)
        existing = table.schema.metadata or {}
        meta = dict(existing)
        meta[_PEAK_EQUITY_META_KEY] = repr(float(self.peak_equity)).encode("utf-8")
        return table.replace_schema_metadata(meta)

    def save(self, path: PathLike) -> None:
        """Write the ledger to a local parquet file (peak_equity in metadata)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(self._to_arrow_table(), str(path))

    def to_bytes(self) -> bytes:
        """Serialize the ledger to in-memory parquet bytes."""
        buf = io.BytesIO()
        pq.write_table(self._to_arrow_table(), buf)
        return buf.getvalue()

    @classmethod
    def _from_arrow_table(cls, table: pa.Table) -> "PositionLedger":
        peak_equity = 0.0
        meta = table.schema.metadata or {}
        if _PEAK_EQUITY_META_KEY in meta:
            try:
                peak_equity = float(meta[_PEAK_EQUITY_META_KEY].decode("utf-8"))
            except (ValueError, AttributeError):
                peak_equity = 0.0
        df = table.to_pandas()
        return cls(df=df, peak_equity=peak_equity)

    @classmethod
    def load(cls, path: PathLike) -> "PositionLedger":
        """Load from a local parquet path. Missing path -> empty ledger."""
        path = Path(path)
        if not path.exists():
            return cls()
        table = pq.read_table(str(path))
        return cls._from_arrow_table(table)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "PositionLedger":
        """Load from in-memory parquet bytes."""
        table = pq.read_table(io.BytesIO(raw))
        return cls._from_arrow_table(table)
