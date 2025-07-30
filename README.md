# CleanInsider

CleanInsider is a data pipeline that consolidates insider trading activity with market and fundamental metrics. The project scrapes multiple public sources, engineers features and outputs a machine‑learning ready dataset.

## Overview

The pipeline collects insider purchase data from **OpenInsider**, fundamental information from **SEC EDGAR** filings, OHLCV price history from **Stooq**, and several U.S. macroeconomic indicators. Each module runs in parallel using **joblib** and displays progress with **tqdm**. Data is merged and post‑processed into a tidy feature table.

Key stages include:

1. **OpenInsider scraping** – weekly insider trades with role parsing and daily aggregation.
2. **SEC EDGAR lookup** – recent quarterly fundamentals with engineered ratios.
3. **Technical indicators** – moving averages, momentum and volatility features computed from Stooq price data.
4. **Macro features** – selected macroeconomic series matched to filing dates.
5. **Feature preprocessing** – remove highly missing or correlated columns, clip outliers and create composite metrics.

The entry point `scrape_data.py` orchestrates scraping and preprocessing in one command.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default paths for input datasets and outputs are defined in `src/config.py`. Adjust them if your local data lives elsewhere.

## Usage

Run the full pipeline for the latest three weeks of OpenInsider activity:

```bash
python scrape_data.py --weeks 3
```

The script creates a `data/` directory (ignored by Git) and saves both the raw merged features and the final preprocessed set in `data/scrapers/features/`.

## Repository Structure

- `src/` – scraping modules, preprocessing utilities and configuration
- `scrape_data.py` – CLI wrapper that runs the pipeline end to end
- `requirements.txt` – Python dependencies
- `pyproject.toml` – formatting settings for **black** and **ruff**

## Development

The project uses **ruff** and **black** for linting and formatting. You can run them locally before committing:

```bash
ruff check src
black src
```

GitHub Actions automatically runs these tools on pull requests and pushes to `main`.

## License

This repository is provided without a license. All rights reserved by the author.
