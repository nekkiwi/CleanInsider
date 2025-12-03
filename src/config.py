import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Add this at the top of config.py

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

MASTER_EVENT_LIST_PATH = DATA_DIR / "scrapers" / "targets" / "master_event_list.parquet"
MODELS_PATH = DATA_DIR / "models"

# --- Scraping Parameters ---
# User agent for making polite requests
REQUESTS_HEADER = {"User-Agent": "YourName/YourProject my.email@domain.com"}

# List of all output directories to be created by the pipeline
DIRECTORIES_TO_CREATE = [FEATURES_OUTPUT_PATH, FEATURES_INFO_OUTPUT_PATH]

# --- Alpaca Trading Configuration ---
# API keys are loaded from environment variables for security
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

# Paper trading mode: True for paper, False for live
PAPER_MODE = os.environ.get("PAPER_MODE", "true").lower() == "true"

# Alpaca API endpoints
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"

# --- Google Drive Configuration ---
# Service account credentials JSON (base64 encoded in env for GitHub Actions)
GOOGLE_DRIVE_CREDENTIALS = os.environ.get("GOOGLE_DRIVE_CREDENTIALS", "")

# Folder IDs on Google Drive
GDRIVE_MODELS_FOLDER_ID = os.environ.get("GDRIVE_MODELS_FOLDER_ID", "")

# Google Sheets ID for logging (spreadsheet ID, not folder)
GDRIVE_LOG_SHEET_ID = os.environ.get("GDRIVE_LOG_SHEET_ID", os.environ.get("GDRIVE_LOGS_FOLDER_ID", ""))

# --- Inference Configuration ---
# Best performing strategy configuration
DEFAULT_STRATEGY = ("1w", 0.05, -0.05)
DEFAULT_THRESHOLD_PCT = 2

# Ensemble configuration: all 5 folds x 5 seeds = 25 models
ENSEMBLE_FOLDS = [1, 2, 3, 4, 5]
ENSEMBLE_SEEDS = [42, 123, 2024, 456, 567]

# Preprocessing artifacts path
PREPROCESSING_ARTIFACTS_PATH = FEATURES_OUTPUT_PATH / "preprocessing"
COMMON_FEATURES_PATH = PREPROCESSING_ARTIFACTS_PATH / "common_features.json"

# --- Position Sizing & Risk Management ---
# Maximum position size as fraction of portfolio
MAX_POSITION_SIZE = 0.05  # 5% max per position

# Maximum total exposure as fraction of portfolio
MAX_TOTAL_EXPOSURE = 0.50  # 50% max total invested

# Minimum position size in dollars
MIN_POSITION_DOLLARS = 100

# Maximum spread cost threshold (positions with higher costs are skipped)
MAX_SPREAD_COST = 0.03  # 300 bps (3%) - higher threshold for small caps

# Ensemble voting threshold (fraction of models that must agree)
ENSEMBLE_VOTE_THRESHOLD = 0.5  # Majority vote

# --- Logging Configuration ---
LOG_DIR = ROOT_DIR / "logs"
TRADE_LOG_PATH = LOG_DIR / "trades"
PERFORMANCE_LOG_PATH = LOG_DIR / "performance"

# --- Data Scraping Settings for Live Inference ---
LIVE_SCRAPE_WEEKS = 2  # Number of weeks to scrape for live inference
