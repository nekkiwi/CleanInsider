# CleanInsider

CleanInsider is an automated equity-trading system that follows **publicly-disclosed insider purchases**. It scrapes insider buys (OpenInsider), fundamentals (SEC EDGAR), price technicals and macro series (Stooq); engineers a feature table; trains walk-forward LightGBM ensembles on take-profit / stop-loss / holding-horizon outcomes; and runs a scheduled pipeline that sizes positions and trades them on Alpaca (paper by default).

> For a detailed map of the architecture and conventions, see **`CLAUDE.md`**.

## Pipeline overview

```
scrape_data.py        →  insider/fundamental/technical/macro features + preprocessing + targets   (data/scrapers/…)
train_walk_forward.py →  walk-forward LightGBM classifier+regressor ensembles                      (data/models/…)
prepare_deploy.py     →  flatten selected models for Google Drive upload                            (deploy/…)
run_inference.py      →  scrape live events → predict → size → trade on Alpaca → log               (logs/…, Sheets)
```

Five stages:
1. **Feature scraping** (`src/scrapers/feature_scraper/`): OpenInsider trades (role parsing, daily aggregation), SEC EDGAR fundamentals, Stooq technical indicators, macro series.
2. **Preprocessing** (`src/preprocess/`): a two-pass, time-based walk-forward split (N validation folds + a held-out test set). Pass 1 learns drop rules (correlation/variance/missing) per fold and the intersection of surviving features (`common_features.json`); pass 2 transforms and writes per-fold datasets.
3. **Targets** (`src/scrapers/target_scraper/`): for each event, label = enter at the first trading day ≥ filing date, exit at the first of take-profit / stop-loss / horizon end (1w=5, 2w=10, 1m=21 business days); return is expressed as **alpha vs SPX**. Corwin–Schultz spread estimates are attached as a trading-cost feature.
4. **Training** (`src/training_pipeline.py`): per (strategy, seed, fold), a LightGBM classifier (return ≥ threshold?) and a regressor (continuous return, positives only). Saved to `data/models/{strategy}/fold_{f}/seed_{s}/`.
5. **Inference & trading** (`src/alpaca/`): an ensemble (folds × seeds) votes for buy signals; positions are sized and executed via Alpaca; results are logged to Google Sheets.

## Installation

```powershell
# Windows / PowerShell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python **3.11** is the supported version (see `.python-version`; CI runs 3.11). Paths and parameters are centralized in `src/config.py`; secrets are loaded from a local `.env` (see below). The `data/` directory is git-ignored and not distributed.

### Environment / secrets (`.env`)

| Variable | Purpose |
|---|---|
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Alpaca trading + market data |
| `PAPER_MODE` | `true` (paper) / `false` (live) — defaults to paper |
| `GOOGLE_DRIVE_CREDENTIALS` | Service-account JSON: base64, raw, or a path to `service_account.json` |
| `GDRIVE_MODELS_FOLDER_ID` | Drive folder holding deployed model weights |
| `GDRIVE_LOG_SHEET_ID` | Google Sheet for trade/performance logging (falls back to `GDRIVE_LOGS_FOLDER_ID`) |

## Common commands

```powershell
# Lint / format (config in pyproject.toml)
python -m ruff check .
python -m black --check .

# Tests
python -m pytest tests/ -q                              # full suite (needs models + creds for slow/integration)
python -m pytest tests/ -q -m "not slow and not integration"   # CI-mode: fast unit tests only
python tests/run_tests.py                               # fast standalone runner (no pytest)

# Pipelines (run from repo root)
python scrape_data.py --weeks 3        # scrape + preprocess + targets
python train_walk_forward.py           # walk-forward training
python run_inference.py --no-trade     # live scrape + predict, no orders
python run_inference.py --dry-run      # size positions, no orders
```

`pytest` markers: `slow` (needs trained models/data on disk) and `integration` (hits live Alpaca/Google APIs). CI deselects both. The files `tests/test_e2e.py`, `tests/test_logging.py`, `tests/quick_alpaca_test.py`, and `tests/debug_inference.py` are **manual diagnostic scripts**, run directly with `python tests/<file>.py`.

## Continuous integration

`.github/workflows/ci.yaml` runs ruff, black `--check`, and the CI-mode test subset on every PR and push to `main` (no secrets, no trading). The daily trading workflow lives in `.github/workflows/` and runs LightGBM inference on Alpaca paper.

## Status & roadmap

This repository is undergoing an upgrade to a fully-automated paper→live bot: volatility-targeted position sizing, take-profit/stop-loss bracket orders with horizon-based exits, a Drive-backed position ledger, split entry/reconcile workflows, and a TabPFN3 research benchmark. See the staged plan tracked alongside the work.

## License

Provided without a license; all rights reserved by the author. Note: any TabPFN model weights used for research are under a separate non-commercial license and are not used for real-money trading.
