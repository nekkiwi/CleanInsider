from pathlib import Path

# --- Base Directories ---
# Resolves the project's root directory dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# --- Output Data Paths ---
FEATURES_OUTPUT_PATH = DATA_DIR / "scrapers" / "features"
FEATURES_INFO_OUTPUT_PATH = FEATURES_OUTPUT_PATH / "info"
EDGAR_DOWNLOAD_PATH = DATA_DIR / "sec_database" / "parquet"
STOOQ_DATABASE_PATH = DATA_DIR / "stooq_database"

RAW_FEATURES_PATH = FEATURES_OUTPUT_PATH / "raw_features.parquet"
PREPROCESSED_FEATURES_PATH = FEATURES_OUTPUT_PATH / "preprocessed_features.parquet"
TARGETS_OUTPUT_PATH = DATA_DIR / "scrapers" / "targets"

# --- Scraping Parameters ---
# User agent for making polite requests
REQUESTS_HEADER = {"User-Agent": "YourName/YourProject my.email@domain.com"}

# List of all output directories to be created by the pipeline
DIRECTORIES_TO_CREATE = [FEATURES_OUTPUT_PATH, FEATURES_INFO_OUTPUT_PATH]
