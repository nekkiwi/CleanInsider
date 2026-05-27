import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Add this at the top of config.py

# Force UTF-8 stdout/stderr so the pipeline's emoji log markers (✅/❌/🚀) don't
# crash with UnicodeEncodeError on Windows' default cp1252 console. config is the
# universal import for every pipeline process, so this applies everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # already UTF-8, or non-reconfigurable
        pass

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
# User agent for making polite requests. SEC EDGAR requires a descriptive
# User-Agent with a contact email; set SEC_USER_AGENT in .env to override.
REQUESTS_HEADER = {
    "User-Agent": os.environ.get(
        "SEC_USER_AGENT", "CleanInsider research nekki.wi@gmail.com"
    )
}

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
GDRIVE_LOG_SHEET_ID = os.environ.get(
    "GDRIVE_LOG_SHEET_ID", os.environ.get("GDRIVE_LOGS_FOLDER_ID", "")
)

# --- Inference Configuration ---
# Best performing strategy configuration
DEFAULT_STRATEGY = ("1w", 0.05, -0.05)
DEFAULT_THRESHOLD_PCT = 2

# Ensemble configuration: all 5 folds x 5 seeds = 25 models
ENSEMBLE_FOLDS = [1, 2, 3, 4, 5]
ENSEMBLE_SEEDS = [42, 123, 2024, 456, 567]

# --- Strategy grid (single source of truth) ---
# (timepoint, take_profit, stop_loss). Used by both target generation
# (scrape_data.py) and training (train_walk_forward.py) so they never drift.
# This is the union of strategies previously trained and deployed.
STRATEGY_GRID = [
    ("1w", 0.05, -0.05),
    ("1w", 0.05, -0.10),
    ("1w", 0.10, -0.05),
    ("1w", 0.10, -0.10),
    ("1w", 0.15, -0.05),
    ("1w", 0.15, -0.10),
    ("2w", 0.05, -0.05),
    ("2w", 0.05, -0.10),
    ("2w", 0.10, -0.10),
    ("1m", 0.05, -0.05),
    ("1m", 0.05, -0.10),
    ("1m", 0.10, -0.10),
]

# Number of walk-forward validation folds (a held-out test set is added on top).
NUM_VALIDATION_FOLDS = 5
# Binary classification thresholds (percent alpha) to label/train against.
BINARY_THRESHOLDS_PCT = [2]
# Number of top features selected per fold during training.
TOP_N_FEATURES = 100

# Research-only: max training rows fed to TabPFN per (strategy, fold, seed).
# TabPFN is an in-context transformer whose memory/latency scale with the
# training context; it is also designed for modest table sizes. Seed-subsample
# training rows above this cap (val/test are never subsampled). LightGBM is
# unaffected. Tune down (e.g. 5000) if the GTX 1080 Ti (11 GB) OOMs.
TABPFN_MAX_ROWS = 1000000

# Feature-ablation hook: comma-separated column-name prefixes to EXCLUDE from
# the feature matrix (in addition to Ticker/Filing Date/alpha_*).
# DEFAULT DROPS FUNDAMENTALS (FIN_/FE_): the yfinance fundamentals path leaks
# the future (same snapshot copied to every filing date for a ticker — proven
# 2026-05-27), and an ablation showed they add ~no edge (Sharpe 2.52->2.39).
# Dropping them here removes the leak at training time without a re-scrape.
# Override with the env var to re-enable once a point-in-time SEC source exists.
DROP_FEATURE_PREFIXES = tuple(
    p.strip()
    for p in os.getenv("DROP_FEATURE_PREFIXES", "FIN_,FE_").split(",")
    if p.strip()
)

# Net-of-cost target: train classifier/regressor on (gross alpha - round-trip
# Corwin-Schultz spread) and label net >= 0, dropping events with no spread.
# Aligns the objective with CAPTURABLE return so the model learns (via the
# spread feature) to avoid high-spread illiquid names instead of chasing gross
# alpha into untradeable micro-caps. Set to "false" to train on gross alpha.
NET_OF_COST_TARGET = os.getenv("NET_OF_COST_TARGET", "true").lower() == "true"


def strategy_target_combinations():
    """STRATEGY_GRID as the {time, tp, sl} dicts expected by target generation."""
    return [{"time": t, "tp": tp, "sl": sl} for (t, tp, sl) in STRATEGY_GRID]


# Preprocessing artifacts path
PREPROCESSING_ARTIFACTS_PATH = FEATURES_OUTPUT_PATH / "preprocessing"
COMMON_FEATURES_PATH = PREPROCESSING_ARTIFACTS_PATH / "common_features.json"

# --- Position Sizing & Risk Management ---
# Maximum position size as fraction of portfolio
MAX_POSITION_SIZE = 0.05  # 5% max per position

# Maximum total exposure as fraction of portfolio
MAX_TOTAL_EXPOSURE = 3.0  # 300% for paper trading (allows margin)

# Minimum position size in dollars
MIN_POSITION_DOLLARS = 100

# Maximum spread cost threshold (positions with higher costs are skipped)
MAX_SPREAD_COST = 0.03  # 300 bps (3%) - higher threshold for small caps

# Ensemble voting threshold (fraction of models that must agree)
ENSEMBLE_VOTE_THRESHOLD = 0.5  # Majority vote

# --- Volatility-targeted sizing (Stage 4) ---
# Sizing method: "vol_target" (ATR risk-budget) or "minmax" (legacy conviction
# fraction of MAX_POSITION_SIZE). vol_target is the default going forward;
# "minmax" preserves the original calculate_dollar_sizes behavior for parity.
SIZING_METHOD = os.environ.get("SIZING_METHOD", "vol_target")

# Fraction of portfolio risked per trade (loss at the stop). 1.25% per name.
RISK_PER_TRADE_PCT = 0.0125

# Target aggregate net long exposure as a fraction of portfolio value. The
# vol_target batch step scales all allocations down so that
# current_exposure + sum(new) does not exceed this.
TARGET_NET_EXPOSURE = 0.50  # 50% net long

# ATR (Average True Range) lookback in daily bars, and the stop multiple used to
# turn ATR into a per-share stop distance. stop_distance = ATR_STOP_MULT * ATR.
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5

# Max tolerated portfolio drawdown before de-risking (reserved for Stage 5 use).
MAX_PORTFOLIO_DD = 0.15

# Upper bound on the fractional-Kelly multiplier (reserved for Stage 5, when the
# realized-trade ledger exists). Kelly cap is a no-op pass-through until then.
KELLY_FRACTION_CAP = 0.5

# --- Logging Configuration ---
LOG_DIR = ROOT_DIR / "logs"
TRADE_LOG_PATH = LOG_DIR / "trades"
PERFORMANCE_LOG_PATH = LOG_DIR / "performance"

# --- Data Scraping Settings for Live Inference ---
LIVE_SCRAPE_WEEKS = 2  # Number of weeks to scrape for live inference

# OHLCV source preference. With a complete local Stooq DB present, prefer it
# (fast, offline, no rate limits) for the bulk historical scrape; fall back to
# yfinance for tickers missing locally. Live CI runs without the Stooq DB simply
# fall through to yfinance. Override with PREFER_LOCAL_OHLCV=false.
PREFER_LOCAL_OHLCV = os.environ.get("PREFER_LOCAL_OHLCV", "true").lower() == "true"

# Bulk-scrape speed switch. When true, OHLCV loading uses local Stooq ONLY (no
# yfinance fallback) — tickers missing from Stooq (mostly delisted) drop instantly
# instead of each incurring a ~15s yfinance timeout + rate-limit, which dominates a
# full scrape. Set OHLCV_LOCAL_ONLY=true for bulk scraping; leave false for live
# inference so the small recent-ticker set can still fall back to yfinance.
OHLCV_LOCAL_ONLY = os.environ.get("OHLCV_LOCAL_ONLY", "false").lower() == "true"
