# file: evaluate_strategies.py
"""Evaluate trained strategies against the success criteria and pick the best.

Reads the walk-forward test-set metrics workbook produced by
``train_walk_forward.py`` and writes ``results/StrategySelection_Report.xlsx``.

Usage:
    python evaluate_strategies.py
    python evaluate_strategies.py --metrics results/LightGBM_Test_Metrics_Summary.xlsx
"""

import argparse
from pathlib import Path

from src.training.strategy_selector import (
    evaluate_strategies,
    select_best,
    write_report,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_METRICS = ROOT / "results" / "LightGBM_Test_Metrics_Summary.xlsx"
DEFAULT_OUT = ROOT / "results" / "StrategySelection_Report.xlsx"


def main(metrics_path: Path, out_path: Path) -> None:
    if not metrics_path.exists():
        raise SystemExit(
            f"Metrics workbook not found: {metrics_path}\n"
            "Run `python train_walk_forward.py` first."
        )

    report = evaluate_strategies(metrics_path)
    best = select_best(report)
    write_report(report, out_path, best=best)

    n_pass = int(report["Pass"].sum()) if not report.empty else 0
    print(f"Evaluated {len(report)} strategies; {n_pass} passed all criteria.")
    print(f"Report written to {out_path}")
    if best is None:
        print("No strategy cleared the success thresholds.")
    else:
        print(
            f"BEST: {best['Strategy']} | "
            f"PortSharpe={best['Portfolio Sharpe (Net)']:.3f} "
            f"PortMaxDD={best['Portfolio Max Drawdown']:.3f} "
            f"PF={best['Profit Factor']:.3f} "
            f"MeanAlpha={best['Mean Alpha (Net)']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select the best strategy by the success criteria."
    )
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.metrics, args.out)
