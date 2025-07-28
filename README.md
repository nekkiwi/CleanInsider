# Quantitative Finance Feature Scraping Pipeline

This project contains a Python-based pipeline designed to scrape, process, and consolidate financial data for quantitative analysis. It gathers insider trading information from OpenInsider, enriches it with technical indicators and financial ratios, and produces a clean, merged dataset ready for feature engineering and modeling.

## Features

- **Modular Scrapers**: Separate, single-responsibility scripts for each data source (OpenInsider, Technical Indicators, Financial Ratios).
- **Robust Fallback Logic**: Prioritizes fast, local data sources (Stooq) and official filings (EDGAR), with fallbacks to APIs (Yahoo Finance) if needed.
- **Centralized Configuration**: All file paths and settings are managed in a central `config.py` file for easy modification.
- **Dependency Management**: A `requirements.txt` file ensures a reproducible environment.
- **Testing Suite**: Uses `pytest` to ensure code correctness and reliability through unit and integration tests.
- **Code Quality Enforcement**: Configured with `ruff` for linting and `black` for consistent code formatting.

## Directory Structure

- `/data/`: Intended location for all scraped data and external datasets (e.g., Stooq). Ignored by Git.
- `/src/`: Contains all the core source code for the scraping pipeline.
- `/tests/`: Contains all unit and integration tests.
- `scrape_data.py`: The main executable script to run the entire pipeline.
- `requirements.txt`: A list of all Python packages required for this project.
- `pyproject.toml`: Configuration file for development tools like `black` and `ruff`.

## Setup Instructions

**1. Clone the Repository**
git clone <your-repo-url>
cd quantitative-finance-pipeline

**2. Create and Activate a Virtual Environment**
It is highly recommended to use a virtual environment to manage project-specific dependencies.