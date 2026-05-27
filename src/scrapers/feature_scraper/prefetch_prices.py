# file: src/scrapers/feature_scraper/prefetch_prices.py
"""
Recover price history for tickers absent from the local Stooq DB (mostly delisted
names) by batch-downloading them from yfinance and caching them as Stooq-format
.txt files. Batched downloads avoid the per-ticker rate-limiting that makes a
one-at-a-time yfinance fallback prohibitively slow during a full scrape.

Run once before technicals/targets so the rest of the pipeline can read prices
local-only (OHLCV_LOCAL_ONLY=true) — fast, and with maximal survivorship coverage.
"""

import warnings
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf

    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

RECOVERED_SUBDIR = ("us", "yf_recovered")
STOOQ_HEADER = (
    "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"
)


def _existing_stooq_symbols(db_path: Path) -> set:
    """All ticker symbols already present locally (first dot-segment of each .txt)."""
    return {f.name.lower().split(".")[0] for f in db_path.rglob("*.txt")}


def _write_stooq_txt(ticker: str, df: pd.DataFrame, out_dir: Path) -> bool:
    """Write a yfinance OHLCV frame as a Stooq-format .txt. Returns True if written."""
    tk_u = ticker.upper()
    lines = [STOOQ_HEADER]
    for dt, row in df.iterrows():
        try:
            o, h, low, c = (
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if pd.isna(c):
            continue
        vol = row.get("Volume")
        vol = int(vol) if pd.notna(vol) else 0
        lines.append(
            f"{tk_u}.US,D,{pd.Timestamp(dt).strftime('%Y%m%d')},000000,"
            f"{o},{h},{low},{c},{vol},0"
        )
    if len(lines) <= 21:  # header + at least 20 rows of data
        return False
    (out_dir / f"{ticker.lower()}.us.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return True


def prefetch_missing_tickers(
    tickers,
    stooq_db_path,
    batch_size: int = 100,
    start: str = "2009-01-01",
) -> int:
    """
    Batch-download tickers absent from the local Stooq DB and cache them locally.

    Args:
        tickers: iterable of ticker symbols (e.g. the OpenInsider event universe).
        stooq_db_path: root of the local Stooq database.
        batch_size: tickers per yfinance batch request (batching avoids rate limits).
        start: history start date.

    Returns:
        Number of tickers recovered and written to the local cache.
    """
    if not YF_AVAILABLE:
        print("[PREFETCH] yfinance unavailable; skipping delisted-ticker recovery.")
        return 0

    db = Path(stooq_db_path)
    existing = _existing_stooq_symbols(db)

    def present(t: str) -> bool:
        tl = t.lower()
        return tl in existing or tl.replace(".", "-") in existing

    missing = sorted({t for t in tickers if t and not present(t)})
    if not missing:
        print("[PREFETCH] local Stooq covers all event tickers; nothing to fetch.")
        return 0

    out_dir = db.joinpath(*RECOVERED_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_batches = (len(missing) + batch_size - 1) // batch_size
    print(
        f"[PREFETCH] {len(missing)} tickers absent from Stooq; "
        f"batch-fetching from yfinance in {n_batches} batches..."
    )

    recovered = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for bi in range(0, len(missing), batch_size):
            batch = missing[bi : bi + batch_size]
            try:
                data = yf.download(
                    batch,
                    start=start,
                    progress=False,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                )
            except Exception as e:
                print(f"[PREFETCH] batch {bi // batch_size + 1} download failed: {e}")
                continue
            for tk in batch:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if tk not in data.columns.get_level_values(0):
                            continue
                        sub = data[tk]
                    else:
                        sub = data
                    sub = sub.dropna(how="all")
                except (KeyError, TypeError):
                    continue
                if sub is None or sub.empty:
                    continue
                if _write_stooq_txt(tk, sub, out_dir):
                    recovered += 1
            print(
                f"[PREFETCH]   batch {bi // batch_size + 1}/{n_batches}: "
                f"{recovered} recovered so far"
            )

    print(
        f"[PREFETCH] recovered {recovered}/{len(missing)} absent tickers -> {out_dir}"
    )
    return recovered
