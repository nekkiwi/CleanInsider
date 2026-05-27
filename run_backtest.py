# run_backtest.py
"""
Daily mark-to-market ensemble backtest entry point.

Scores every strategy in ``config.STRATEGY_GRID`` on the held-out test set with
a TRUSTWORTHY portfolio Sharpe / MaxDD (genuine daily equity curve, net of a
one-way Corwin-Schultz spread cost), in contrast to the inflated per-trade
Sharpe (~2.4) the training summaries report.

Outputs ``results/{model_type}_Backtest_Metrics.xlsx`` and prints the best
PASSING strategy by net portfolio Sharpe.

Usage:
    python run_backtest.py
    python run_backtest.py --model-type LightGBM
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.training import strategy_selector
from src.training.ensemble_backtest import run_all

SPX_TICKER_LOCAL = "^spx"


def build_spx_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Load the local Stooq SPX parquet into (dates, close) numpy arrays.

    Mirrors how target generation builds ``spx_arrays`` (see
    calculate_master_targets / generate_targets): index -> datetime64[ns],
    Close -> float array.
    """
    spx_path = Path(config.STOOQ_DATABASE_PATH) / f"{SPX_TICKER_LOCAL}.parquet"
    if not spx_path.exists():
        raise FileNotFoundError(
            f"SPX benchmark not found at {spx_path}. The daily backtest needs the "
            f"local Stooq ^spx index to compute alpha vs SPX."
        )
    spx = pd.read_parquet(spx_path)
    if spx.index.tz is not None:
        spx.index = spx.index.tz_localize(None)
    return (
        spx.index.to_numpy(dtype="datetime64[ns]"),
        spx["Close"].to_numpy(dtype=float),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily-MTM ensemble backtest.")
    parser.add_argument(
        "--model-type",
        default="LightGBM",
        help="Model family label; selects the output workbook name.",
    )
    args = parser.parse_args()

    # --- Guard: required inputs must be present (clear error, no traceback) ---
    test_features_path = config.FEATURES_OUTPUT_PATH / "test_set" / "test_data.parquet"
    test_spreads_path = config.TARGETS_OUTPUT_PATH / "test_set" / "test_spreads.parquet"

    # Namespace non-LightGBM families: models live under MODELS_PATH/{model_type}/
    # (mirrors ModelTrainer._model_strategy_dir). LightGBM keeps the root path.
    if args.model_type == "LightGBM":
        models_base_path = config.MODELS_PATH
    else:
        models_base_path = config.MODELS_PATH / args.model_type

    if not models_base_path.exists():
        print(f"[ERROR] Models directory not found: {models_base_path}")
        print("        Train models first (python train_walk_forward.py).")
        return 1
    if not test_features_path.exists():
        print(f"[ERROR] Test feature set not found: {test_features_path}")
        print("        Run the scrape/preprocess pipeline to produce the test set.")
        return 1

    try:
        spx_arrays = build_spx_arrays()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    print(f"\n### DAILY-MTM ENSEMBLE BACKTEST: {args.model_type} ###")
    print(f"  Test features: {test_features_path}")
    print(f"  Test spreads : {test_spreads_path}")
    print(f"  Models       : {models_base_path}")
    print(f"  Strategies   : {len(config.STRATEGY_GRID)}")
    print(
        "  Capital model: net-of-cost; tradeable universe (quoted spread "
        f"<= {config.MAX_SPREAD_COST:.0%}); per-name <= {config.MAX_POSITION_SIZE:.0%}; "
        "gross <= 100% (no leverage); unfilled exposure = cash."
    )

    report = run_all(
        strategies=config.STRATEGY_GRID,
        model_type=args.model_type,
        models_base_path=models_base_path,
        test_features_path=test_features_path,
        test_spreads_path=test_spreads_path,
        spx_arrays=spx_arrays,
        out_dir=config.ROOT_DIR / "results",
        write_xlsx=True,
    )

    if report.empty:
        print("[ERROR] No strategy produced a backtest row (models all missing?).")
        return 1

    selection = strategy_selector.evaluate_backtest(report)
    best = strategy_selector.select_best(selection, metric="sharpe")

    print("\n--- Backtest metrics (net portfolio Sharpe vs inflated per-trade) ---")
    cols = [
        c
        for c in [
            "strategy_str",
            "sharpe",
            "raw_sharpe",
            "max_drawdown",
            "total_alpha",
            "n_trades",
        ]
        if c in selection.columns
    ]
    print(selection[cols + ["Pass"]].to_string(index=False))

    if best is None:
        print(
            "\n[RESULT] No strategy PASSES the success criteria "
            "(sharpe>=0.75, max_drawdown<=0.15, total_alpha>0)."
        )
    else:
        print(
            f"\n[RESULT] Best passing strategy: {best['strategy_str']} "
            f"(net Sharpe {best['sharpe']:.2f}, MaxDD {best['max_drawdown']:.2%}, "
            f"total alpha {best['total_alpha']:.2%})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
