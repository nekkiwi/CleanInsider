from pathlib import Path

# --- Base Directories ---
# Resolves the project's root directory dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# --- Input Data Paths ---
# Path to the root of the Stooq historical database directory structure.
# The loader will recursively search this directory for ticker files.
STOOQ_DATABASE_PATH = DATA_DIR / "historical_stock_db"

# --- Output Data Paths ---
FEATURES_DIR = DATA_DIR / "scrapers" / "features"
EDGAR_DOWNLOAD_PATH = DATA_DIR / "edgar_filings"

# Stage-specific output files
STAGE_1_PATH = FEATURES_DIR / "stage_1_openinsider.xlsx"
STAGE_2_PATH = FEATURES_DIR / "stage_2_technical_indicators.xlsx"
STAGE_3_PATH = FEATURES_DIR / "stage_3_financial_ratios.xlsx"
FINAL_OUTPUT_PATH = FEATURES_DIR / "all_features.xlsx"

# --- Scraping Parameters ---
# User agent for making polite requests
REQUESTS_HEADER = {'User-Agent': 'YourName/YourProject my.email@domain.com'}

# List of all output directories to be created by the pipeline
DIRECTORIES_TO_CREATE = [FEATURES_DIR, EDGAR_DOWNLOAD_PATH]

