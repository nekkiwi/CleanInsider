# run_robustness.py
"""
Robustness / stress harness entry point for the liquid insider strategy.

Mirrors ``run_backtest.py`` (builds ``spx_arrays``, merges entry Price + ADV
onto the held-out test features, defaults to the liquid universe) and then runs
EVERY stress sweep + the multi-regime walk-forward OOS for the top strategies,
writing the tables to ``results/Robustness_Report.xlsx``.

This is the script the human runs AFTER the harness is built/unit-tested. It
loads the REAL 25-model ensembles and the REAL test/validation data, so it is
not part of the unit suite.

Usage:
    python run_robustness.py
    python run_robustness.py --model-type LightGBM
    python run_robustness.py --strategy 1w_tp0p05_sl-0p05 --strategy 2w_tp0p05_sl-0p05
    python run_robustness.py --n-trials 60        # trials for the deflated Sharpe
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import build_spx_arrays
from src import config
from src.training.backtester import simulate_daily_portfolio
from src.training.ensemble_backtest import (
    _ensemble_signals,
    _joblib_dir_loader,
    _load_ensemble,
    backtest_strategy,
    strategy_string,
)
from src.training.robustness import (
    deflated_sharpe,
    sweep_capital,
    sweep_cost,
    sweep_entry_timing,
    sweep_liquidity,
    walk_forward_oos,
)
from src.training.training_helpers import (
    calculate_position_sizes,
    horizon_business_days,
)

# Top liquid strategies to stress (override with --strategy). The 1w 5%/-5% is
# the documented DEFAULT_STRATEGY / liquid baseline.
DEFAULT_STRATEGIES = [
    ("1w", 0.05, -0.05),
    ("2w", 0.05, -0.05),
    ("1m", 0.05, -0.05),
]


def _parse_strategy(s: str) -> tuple:
    """Parse a folder-style id (1w_tp0p05_sl-0p05) back into a strategy tuple."""
    for strat in config.STRATEGY_GRID:
        if strategy_string(strat) == s:
            return strat
    raise ValueError(f"Unknown strategy id {s!r}; not in config.STRATEGY_GRID")


def _merge_price_adv(test_features_df: pd.DataFrame) -> pd.DataFrame:
    """Merge entry Price + ADV onto the test features (mirror run_backtest)."""
    df = test_features_df.copy()
    df["Filing Date"] = pd.to_datetime(df["Filing Date"])
    if "Price" not in df.columns and Path(config.MASTER_EVENT_LIST_PATH).exists():
        mel = pd.read_parquet(config.MASTER_EVENT_LIST_PATH)[
            ["Ticker", "Filing Date", "Price"]
        ].drop_duplicates(["Ticker", "Filing Date"])
        mel["Filing Date"] = pd.to_datetime(mel["Filing Date"])
        df = df.merge(mel, on=["Ticker", "Filing Date"], how="left")
    if "adv" not in df.columns and Path(config.ADV_COMPONENT_PATH).exists():
        adv = pd.read_parquet(config.ADV_COMPONENT_PATH)[
            ["Ticker", "Filing Date", "adv"]
        ].drop_duplicates(["Ticker", "Filing Date"])
        adv["Filing Date"] = pd.to_datetime(adv["Filing Date"])
        df = df.merge(adv, on=["Ticker", "Filing Date"], how="left")
    return df


def _baseline_daily_returns(
    strategy: tuple, models_base_path: Path, test_features_df, spx_arrays
) -> pd.Series:
    """Reconstruct the baseline run's daily port_alpha series.

    The deflated Sharpe needs the actual daily-return series, which
    ``backtest_strategy`` does not return (only summary metrics), so we replay
    the liquid path here using the same building blocks.
    """
    folds = config.ENSEMBLE_FOLDS
    seeds = config.ENSEMBLE_SEEDS
    timepoint, tp, sl = strategy
    horizon_days = horizon_business_days(timepoint)

    models = _load_ensemble(
        strategy, models_base_path, folds, seeds, _joblib_dir_loader
    )
    signals = _ensemble_signals(
        models, test_features_df, config.ENSEMBLE_VOTE_THRESHOLD
    )
    buys = signals[signals["buy_signal"] == 1]
    if buys.empty:
        return pd.Series(dtype=float)

    weights = calculate_position_sizes(buys["conviction"]) * config.MAX_POSITION_SIZE
    positions = pd.DataFrame(
        {
            "Ticker": test_features_df.loc[buys.index, "Ticker"].to_numpy(),
            "entry_date": test_features_df.loc[buys.index, "Filing Date"].to_numpy(),
            "weight": weights.to_numpy(),
        }
    )
    price = pd.to_numeric(
        test_features_df.loc[buys.index, "Price"], errors="coerce"
    ).to_numpy()
    adv = pd.to_numeric(
        test_features_df.loc[buys.index, "adv"], errors="coerce"
    ).to_numpy()
    positions["entry_cost"] = float(getattr(config, "LIQUID_ROUND_TRIP_COST", 0.002))
    keep = (pd.Series(price) >= config.LIQUID_PRICE_MIN) & (
        pd.Series(adv) >= config.LIQUID_ADV_MIN
    )
    positions = positions[keep.fillna(False).to_numpy()].reset_index(drop=True)
    if positions.empty:
        return pd.Series(dtype=float)

    daily = simulate_daily_portfolio(
        positions,
        tp=tp,
        sl=sl,
        horizon_days=horizon_days,
        spx_arrays=spx_arrays,
        per_name_cap=config.MAX_POSITION_SIZE,
        max_gross_exposure=1.0,
    )
    if daily is None or daily.empty or "port_alpha" not in daily:
        return pd.Series(dtype=float)
    return daily["port_alpha"].dropna()


def main() -> int:
    parser = argparse.ArgumentParser(description="Liquid-strategy robustness sweep.")
    parser.add_argument("--model-type", default="LightGBM")
    parser.add_argument(
        "--strategy",
        action="append",
        default=None,
        help="Strategy id (e.g. 1w_tp0p05_sl-0p05); repeatable. Defaults to the "
        "top three liquid strategies.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=len(config.STRATEGY_GRID),
        help="Trial count for the deflated Sharpe (default = STRATEGY_GRID size).",
    )
    args = parser.parse_args()

    strategies = (
        [_parse_strategy(s) for s in args.strategy]
        if args.strategy
        else DEFAULT_STRATEGIES
    )

    if args.model_type == "LightGBM":
        models_base_path = config.MODELS_PATH
    else:
        models_base_path = config.MODELS_PATH / args.model_type

    test_features_path = config.FEATURES_OUTPUT_PATH / "test_set" / "test_data.parquet"
    features_dir = config.FEATURES_OUTPUT_PATH
    targets_dir = config.TARGETS_OUTPUT_PATH

    if not models_base_path.exists():
        print(f"[ERROR] Models directory not found: {models_base_path}")
        return 1
    if not test_features_path.exists():
        print(f"[ERROR] Test feature set not found: {test_features_path}")
        return 1

    try:
        spx_arrays = build_spx_arrays()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    test_features_df = _merge_price_adv(pd.read_parquet(test_features_path))

    print("\n### ROBUSTNESS / STRESS HARNESS ###")
    print(f"  Strategies : {[strategy_string(s) for s in strategies]}")
    print(f"  Models     : {models_base_path}")
    print(f"  Test feat  : {test_features_path}")
    print(f"  Deflate N  : {args.n_trials} trials\n")

    bt_common = dict(
        models_base_path=models_base_path,
        test_features_df=test_features_df,
        spx_arrays=spx_arrays,
    )

    sheets: dict[str, pd.DataFrame] = {}
    summary_rows = []

    for strategy in strategies:
        sid = strategy_string(strategy)
        print(f"--- {sid} ---")

        baseline = backtest_strategy(strategy, liquid_mode=True, **bt_common)
        print(
            f"  baseline: net Sharpe {baseline['sharpe']:.2f}, "
            f"total alpha {baseline['total_alpha']:.2%}, "
            f"n_trades {baseline['n_trades']}"
        )

        entry = sweep_entry_timing(strategy, liquid_mode=True, **bt_common)
        cost = sweep_cost(strategy, **bt_common)
        liq = sweep_liquidity(strategy, **bt_common)
        cap = sweep_capital(strategy, liquid_mode=True, **bt_common)
        wfo = walk_forward_oos(
            strategy,
            models_base_path=models_base_path,
            features_dir=features_dir,
            targets_dir=targets_dir,
            spx_arrays=spx_arrays,
            ohlcv_loader=_default_ohlcv_loader_for(strategy),
        )

        daily = _baseline_daily_returns(
            strategy, models_base_path, test_features_df, spx_arrays
        )
        dsr = (
            deflated_sharpe(daily.to_numpy(), n_trials=args.n_trials)
            if not daily.empty
            else float("nan")
        )
        print(f"  deflated Sharpe Prob(SR>0): {dsr:.4f}")

        for name, frame in (
            ("entry_timing", entry),
            ("cost", cost),
            ("liquidity", liq),
            ("capital", cap),
            ("walk_forward_oos", wfo),
        ):
            sheets[f"{sid}_{name}"] = frame

        be = cost[cost["break_even"]]
        break_even_bps = float(be["cost_bps"].iloc[0]) if not be.empty else np.nan
        summary_rows.append(
            {
                "strategy": sid,
                "baseline_sharpe": baseline["sharpe"],
                "baseline_total_alpha": baseline["total_alpha"],
                "n_trades": baseline["n_trades"],
                "break_even_cost_bps": break_even_bps,
                "deflated_sharpe_prob": dsr,
            }
        )

    summary = pd.DataFrame(summary_rows)
    out_dir = config.ROOT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Robustness_Report.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        for name, frame in sheets.items():
            # Excel sheet names cap at 31 chars.
            frame.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"\n[INFO] Wrote {out_path}")
    print("\n--- Summary ---")
    print(summary.to_string(index=False))
    return 0


def _default_ohlcv_loader_for(strategy):
    """The real Stooq/yfinance loader (deferred import; entry-point only)."""
    from src.training.backtester import _default_ohlcv_loader

    return _default_ohlcv_loader


if __name__ == "__main__":
    sys.exit(main())
