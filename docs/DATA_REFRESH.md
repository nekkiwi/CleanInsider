# Data Refresh Runbook (Stage 1)

The training data was last current to **2025-05-09** (~1 year stale). This runbook
brings every data source up to date so the models can be retrained (Stage 2).

Run everything from the repo root with the project venv active. The pipeline is
long-running and network-heavy — run it **locally**, not in CI.

## 0. Prerequisites (one-time)

- `.venv` on Python 3.11 with `pip install -r requirements.txt`.
- In `.env`, set a real SEC contact (SEC requires a descriptive User-Agent):
  ```
  SEC_USER_AGENT=CleanInsider research your.email@domain.com
  ```

## 1. Refresh price data — Stooq (REQUIRED, manual)

Technical indicators **and** the target TP/SL simulation read Stooq daily bars,
so prices must extend through the most recent holding window.

1. Download the latest bulk daily data from <https://stooq.com/db/h/>
   (US stocks + ETFs, World, and Macro).
2. Extract into `data/stooq_database/` preserving the existing layout
   (`us/…`, `world/…`, `macro/…`, and the `^spx` index file). Overwrite in place.
3. Sanity check: the `^spx` file's last date should be within a few days of today.

## 2. Refresh SEC fundamentals (OPTIONAL, manual)

Fundamentals are quarterly and the feature is "most recent annual statement," so
being a quarter or two behind is acceptable. To update:

```powershell
pip install secfsdstools          # optional dep, not in requirements.txt
python -m src.scrapers.feature_scraper.download_sec_data
```

This downloads new quarterly zips into `data/sec_database/parquet/quarter/`
(currently latest = `2025q2`). Alternatively, drop newer quarter zips in manually.

## 3. Run the full scrape + preprocess + targets (REQUIRED, long-running)

`scrape_data.py` re-scrapes OpenInsider, regenerates all feature components,
preprocesses into walk-forward folds, and generates targets + spreads for the
**12 strategies** defined in `config.STRATEGY_GRID`.

`--weeks` is how many weeks of OpenInsider history to scrape, counting back from
today. The existing dataset starts **2010-01-08**; to reproduce that full history:

```
weeks ≈ (today − 2010-01-08) / 7  ≈  855   (as of 2026-05)
```

```powershell
# Full refresh (fresh scrape of all components, then preprocess + targets):
python scrape_data.py --weeks 860
```

Notes:
- This **overwrites** `data/scrapers/features/components/*` with a fresh scrape
  (no append/dedup), then rebuilds folds and targets — a complete, clean rebuild.
- Expect this to take **a while** (OpenInsider pages + SEC fundamental lookups for
  thousands of tickers are the slow parts). The target calculation
  (`calculate_master_targets`) is batched and **restartable** if interrupted.
- To only regenerate targets/spreads from existing features (fast path), the
  orchestrators accept `rescrape=False` / `recompute=False` — but for this refresh
  you want the full run above.

## 4. Verify freshness

```powershell
python scripts/check_data_freshness.py --max-stale-days 14
```

Expect `Result: PASS` with every artifact's max date within ~2 weeks of today,
and the test set covering roughly the most recent ~18 months. Exit code 0 = good.

## 5. Hand off to Stage 2

Once fresh, proceed to **retraining + model selection** (Stage 2):
```powershell
python train_walk_forward.py
python evaluate_strategies.py     # produced in Stage 2
```

---

### What changed in the code to enable this (Stage 1)

- `src/scrape_features.py` / `src/scrape_targets.py`: the scrape/calc steps that were
  commented out are now active (gated by `rescrape` / `recompute`, default `True`).
- `src/config.py`: single-source `STRATEGY_GRID` (12 strategies) used by both target
  generation and training; env-driven SEC `User-Agent`; forced UTF-8 stdout so the
  pipeline's emoji log markers no longer crash on Windows (cp1252).
- `scripts/check_data_freshness.py`: freshness report + CI gate.
