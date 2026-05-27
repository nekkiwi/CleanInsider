# file: src/scrapers/feature_scraper/download_sec_data.py
"""
Refresh the local SEC EDGAR financial-statement database (Parquet) used for
fundamental features.

Optional/occasional step: SEC fundamentals are quarterly and the feature is
"most recent annual statement", so being a quarter or two behind is tolerable.
Requires the optional `secfsdstools` package (not in requirements.txt):

    pip install secfsdstools
    python -m src.scrapers.feature_scraper.download_sec_data

Paths come from src.config; the contact email comes from $SEC_USER_AGENT_EMAIL
(falling back to the email embedded in $SEC_USER_AGENT / config.REQUESTS_HEADER).
"""

import os
import re

from src import config


def _contact_email() -> str:
    """Best-effort extract a contact email for SEC's required user-agent."""
    explicit = os.environ.get("SEC_USER_AGENT_EMAIL")
    if explicit:
        return explicit
    ua = config.REQUESTS_HEADER.get("User-Agent", "")
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", ua)
    return match.group(0) if match else "research@example.com"


def setup_and_update_database():
    """Download and process all SEC filings into the local Parquet database."""
    from secfsdstools.a_config.configmodel import Configuration
    from secfsdstools.update import update

    sec_root = config.DATA_DIR / "sec_database"
    db_path = sec_root / "database"
    dld_path = sec_root / "downloads"
    parquet_path = config.EDGAR_DOWNLOAD_PATH  # data/sec_database/parquet

    print("--- Local SEC Database Setup ---")
    print(f"Index/JSONs (db_dir):      {db_path}")
    print(f"ZIP downloads (dld_dir):   {dld_path}")
    print(f"Fast Parquet (parquet_dir):{parquet_path}")
    print(f"Contact email:             {_contact_email()}")
    print("-" * 30)

    sec_config = Configuration(
        db_dir=str(db_path),
        download_dir=str(dld_path),
        parquet_dir=str(parquet_path),
        user_agent_email=_contact_email(),
    )

    print("Starting the download and processing (this can take a long time)...")
    update(sec_config)
    print("\nSEC database download and processing complete.")


if __name__ == "__main__":
    setup_and_update_database()
