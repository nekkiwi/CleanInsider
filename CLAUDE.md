# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

CleanInsider is an end-to-end pipeline that turns **insider-purchase filings into automated equity trades**. It scrapes insider trades (OpenInsider), fundamentals (SEC EDGAR), technicals + macro (Stooq), engineers a feature table, trains walk-forward LightGBM ensembles, and runs daily inference that sizes and submits orders through Alpaca (paper mode by default). The full pipeline is meant to run unattended via GitHub Actions.

The data flows in four stages, each with its own root-level entry point:

```
scrape_data.py  ──►  features + preprocessing + targets   (data/scrapers/…)
train_walk_forward.py  ──►  LightGBM ensembles            (data/models/…)
scripts/prepare_deploy.py  ──►  flattened models for Drive (deploy/…)
run_inference.py  ──►  live signals + Alpaca trades        (logs/…, Google Sheets)
```

## Environment & common commands

Windows dev machine, PowerShell. Python 3.10/3.11. Virtualenv lives in `.venv`.

```powershell
# Activate venv (Windows)
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Lint & format (config in pyproject.toml: black + ruff, line-length 88)
ruff check src
black src

# Tests
pytest tests/                         # full pytest suite
python tests/run_tests.py             # fast direct runner (no model loading)
python tests/run_tests.py --full      # include the 25-model loading test
pytest tests/test_parity.py           # train-vs-inference feature parity
```

There is no `conftest.py`; tests insert the project root on `sys.path` themselves and import via `from src.alpaca...`. Always run Python from the repo root so the `src` package resolves.

## Pipeline entry points (run from repo root)

| Command | Purpose |
|---|---|
| `python scrape_data.py --weeks 3` | Scrape features, preprocess into folds, generate targets |
| `python scrape_data.py --target_only True` | Skip feature scraping/preprocessing, regenerate targets only |
| `python train_walk_forward.py` | Walk-forward train LightGBM classifier+regressor per (strategy, seed, fold) |
| `python run_inference.py --dry-run` | Scrape live events, predict, size positions — **no orders** |
| `python run_inference.py --no-trade --output signals.parquet` | Signals only, save to file |
| `python run_inference.py --model model_1w_tp5_sl5` | Full live run for one strategy (what CI calls) |
| `python scripts/prepare_deploy.py` | Flatten `data/models/` into `deploy/` for Google Drive upload |

> **Critical gotcha:** `src/scrape_features.py` and `src/scrape_targets.py` have their expensive scraping/calculation steps **commented out by design**. As shipped, `scrape_features.py` only *merges* pre-existing component parquets in `data/scrapers/features/components/`, and `scrape_targets.py` only runs the spread estimator. To do a real scrape you must uncomment the relevant blocks. Don't assume `scrape_data.py` re-fetches everything.

## Central configuration — `src/config.py`

Everything path- and parameter-related is centralized here; **always read/modify config through this module**, don't hardcode paths elsewhere. Loads `.env` via `python-dotenv`. Key conventions encoded here:

- **Strategy tuple** `(timepoint, take_profit, stop_loss)`, e.g. `("1w", 0.05, -0.05)`. `DEFAULT_STRATEGY` is the 1-week 5%/-5% strategy.
- **Strategy → folder string**: `f"{timepoint}_tp{tp}_sl{sl}"` with `.` replaced by `p` → `1w_tp0p05_sl-0p05`. This naming is reproduced independently in `inference.py`, `training_pipeline.py`, and `prepare_deploy.py` — keep them in sync.
- **Ensemble** = `ENSEMBLE_FOLDS [1..5]` × `ENSEMBLE_SEEDS [42,123,2024,456,567]` = 25 models. Buy when ≥ `ENSEMBLE_VOTE_THRESHOLD` (0.5) of classifiers vote 1.
- **Target column naming**: `alpha_{timepoint}_tp{tp}_sl{sl}` (note the `alpha_` prefix — these columns are excluded from feature matrices during training).
- **Risk limits**: `MAX_POSITION_SIZE` 5%, `MAX_TOTAL_EXPOSURE` 300% (margin, for paper), `MIN_POSITION_DOLLARS` $100, `MAX_SPREAD_COST` 3%.
- **Secrets from env**: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `PAPER_MODE`, `GOOGLE_DRIVE_CREDENTIALS`, `GDRIVE_MODELS_FOLDER_ID`, `GDRIVE_LOG_SHEET_ID`.

## Architecture details

### Feature → preprocessing → targets (`src/scrapers`, `src/preprocess`)
- **Feature scraping** (`src/scrapers/feature_scraper/`): OpenInsider trades, SEC EDGAR annual statements, Stooq technical indicators, macro series. Outputs component parquets, then `scrape_features.py` merges them into `raw_features.parquet`.
- **Preprocessing** (`src/preprocess_features.py` + `FoldProcessor`): a **2-pass walk-forward** scheme. Time-based splits = `num_folds + 2` (validation folds + one held-out test set). Pass 1 learns drop rules (correlation > 0.8, variance < 1e-4, missing > 0.6) per fold and computes the **intersection of surviving features** → `common_features.json`. Pass 2 transforms and writes `fold_i/{training,validation}_data.parquet` + the test set. The test set is always transformed with the **largest** training fold's processor to avoid leakage.
- **Targets** (`src/scrapers/target_scraper/`): master event list → per-event TP/SL outcome calc (batched, restartable) → per-fold label files → Corwin-Schultz spread estimates. Spreads are merged into training as `corwin_schultz_spread` (a real trading-cost feature, not just a label).

### Training (`src/training_pipeline.py` — `ModelTrainer`)
For each `(seed, strategy, threshold)` × fold: imputation values are learned **from the training set only** (median), feature selection runs on imputed training data (`top_n=100`), then a LightGBM classifier (binary: return ≥ threshold%) and a regressor (continuous return, fit on positive examples only) are trained. Models save to `data/models/{strategy}/fold_{f}/seed_{s}/{classifier,regressor,metadata}.pkl`. `metadata.pkl` carries `selected_features` and `imputation_values` — inference depends on these.

### Inference & trading (`src/alpaca/`)
- `EnsemblePredictor` (`inference.py`): loads all 25 models, rebuilds the **exact per-model feature matrix** from each model's `selected_features` (missing features filled with that model's training imputation value, else 0), majority-votes for buy signal, averages regressor outputs for predicted return.
- `LiveFeatureGenerator` (`live_features.py`): re-scrapes recent insider events and regenerates the *same* feature set. Injects a placeholder `corwin_schultz_spread = 0.005` because live spreads come from Alpaca quotes at sizing time instead.
- `PositionSizer` (`position_sizer.py`): min-max scales predicted returns to `[0.25, 1.0]`, applies a spread haircut (`reference_spread / half_spread`, capped at 1, zeroed above `MAX_SPREAD_COST`), then enforces dollar position/exposure caps.
- `AlpacaTradingClient` (`trading_client.py`): wraps `alpaca-py`; imports are guarded so the module loads even without the SDK. Uses limit orders (0.1% below mid) by default. Fetches live bid/ask spreads and prices.
- `GoogleDriveClient` (`google_drive.py`): downloads models and logs trades/performance to Google Sheets, keyed by `model_id`.

`run_inference.py` orchestrates: download models → scrape live features → filter tradable → predict → **skip already-held tickers** → size with live spreads → execute (unless `--dry-run`/`--no-trade`) → log.

### Deployment & CI
- `scripts/prepare_deploy.py` flattens `fold_X/seed_Y/*.pkl` into `deploy/{model}/weights/fold{X}_seed{Y}_*.pkl` plus `preprocessing/` (uses fold 5's artifacts). Upload `deploy/` to Google Drive.
- `.github/workflows/daily_inference.yaml`: weekday cron after market open. Matrix of three strategies (`model_1w_tp5_sl5`, `model_2w_tp5_sl5`, `model_1m_tp5_sl5`). Pulls weights from Drive via **rclone**, **reconstructs** the `fold_X/seed_Y/` tree from the flattened filenames, then runs `run_inference.py --model <id>`. `PAPER_MODE` is forced to `true`.

## Conventions & cautions

- Models, data, and artifacts live under `data/` and `deploy/`, which are git-ignored. `.gitignore` also ignores `*.json` and `*.csv` globally — be deliberate if you ever need to commit a JSON.
- `.env` and `service_account.json` at the repo root hold live credentials. They are git-ignored; never read, print, or commit their contents.
- The README is partially stale (references a non-existent `scrape_data.py --weeks` Linux-style workflow and an out-of-date overview). Trust this file and `src/config.py` over the README for the current architecture.
- The strategy-string and ensemble fold/seed lists are duplicated across modules. When changing folds, seeds, or the naming scheme, update `config.py`, `inference.py`, `training_pipeline.py`, `prepare_deploy.py`, and the CI matrix together.
</content>
</invoke>
