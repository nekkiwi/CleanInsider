"""
Ensemble strategy backtest on the held-out test set, scored by a TRUSTWORTHY
daily mark-to-market portfolio Sharpe / MaxDD.

Why this exists
---------------
The per-trade ``Sharpe (Net)`` reported by ``evaluate_fold`` treats each closed
trade as one i.i.d. observation. That measures *signal quality*, not the
*portfolio* an investor would actually live with: it ignores calendar overlap,
position concentration, and the real day-to-day equity path. On this low-hit /
high-payoff TP-SL book it inflates Sharpe to ~2.4.

``backtest_strategy`` instead:
  1. Loads the (ENSEMBLE_FOLDS x ENSEMBLE_SEEDS) ensemble for a strategy
     (reusing the same per-model feature-matrix reconstruction as
     ``src.alpaca.inference.EnsemblePredictor``).
  2. Votes on the held-out test set; for buys, averages regressor outputs into a
     conviction and min-max scales it into a position weight.
  3. Builds a positions table with a round-trip cost (full Corwin-Schultz
     spread = ~half-spread on entry + half on exit) and feeds it to
     ``simulate_daily_portfolio`` to obtain a genuine daily alpha curve.
  4. Computes portfolio metrics on ``port_alpha`` (and reports ``port_raw``
     Sharpe as ``raw_sharpe``).

Capital model
-------------
``simulate_daily_portfolio`` uses a realistic CAPPED TARGET-EXPOSURE model:
conviction is min-max scaled into [0.25, 1.0] and multiplied by
``config.MAX_POSITION_SIZE`` (5%), giving per-name CAPITAL weights of 1.25%-5%.
Each day those weights are capped per-name (5%) and the gross is capped at 100%
(no leverage); unfilled exposure sits in cash. Daily portfolio return is the
SUM of weight_i * daily_ret_i over open names (NOT renormalized), so sparse days
carry a genuine cash drag. By default the tradable universe is filtered to names
with a quoted spread <= ``config.MAX_SPREAD_COST`` (the live liquidity filter).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src import config
from src.training.backtester import (
    _default_ohlcv_loader,
    compute_portfolio_metrics,
    simulate_daily_portfolio,
)
from src.training.training_helpers import (
    calculate_position_sizes,
    horizon_business_days,
)

# Round-trip trading cost (return fraction) charged when a spread is unavailable.
# Round-trip drag = the full quoted spread (pay ~half on entry, ~half on exit).
# Entry-only (one-way) costs flatter the backtest ~2x on Sharpe; round-trip is
# the honest default. The live one-way placeholder is 0.005, so round-trip ~ 0.01.
DEFAULT_ROUND_TRIP_COST = 0.01


def strategy_string(strategy: tuple) -> str:
    """Folder-safe strategy id, matching ModelTrainer._get_strategy_string."""
    timepoint, tp, sl = strategy
    return f"{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"


def _load_ensemble(
    strategy: tuple,
    models_base_path: Path,
    folds: list,
    seeds: list,
    loader: Callable[[Path], tuple],
) -> list:
    """Load classifier/regressor/metadata for every (fold, seed).

    ``loader`` maps a model directory -> (classifier, regressor, metadata) so the
    real joblib path and a test stub share one code path.
    """
    strategy_path = Path(models_base_path) / strategy_string(strategy)
    if not strategy_path.exists():
        raise FileNotFoundError(f"Strategy models not found: {strategy_path}")

    models = []
    for fold in folds:
        for seed in seeds:
            model_dir = strategy_path / f"fold_{fold}" / f"seed_{seed}"
            if not model_dir.exists():
                print(f"[WARN] Model directory not found: {model_dir}")
                continue
            classifier, regressor, metadata = loader(model_dir)
            if classifier is None:
                continue
            models.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "classifier": classifier,
                    "regressor": regressor,
                    "selected_features": metadata.get("selected_features", []),
                    "imputation_values": metadata.get("imputation_values", {}),
                }
            )
    return models


def _joblib_dir_loader(model_dir: Path) -> tuple:
    """Default: load the three pickles from a model directory."""
    import joblib

    classifier_path = model_dir / "classifier.pkl"
    regressor_path = model_dir / "regressor.pkl"
    metadata_path = model_dir / "metadata.pkl"
    classifier = joblib.load(classifier_path) if classifier_path.exists() else None
    regressor = joblib.load(regressor_path) if regressor_path.exists() else None
    metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
    return classifier, regressor, metadata


def _build_model_matrix(features_df: pd.DataFrame, model_info: dict) -> pd.DataFrame:
    """Rebuild a model's exact feature matrix (mirrors EnsemblePredictor.predict).

    For each selected feature: take the column if present, else the model's
    training imputation value, else 0. Then fill any residual NaN with the
    imputation value (or 0) and order columns to match ``selected_features``.
    """
    selected_features = model_info["selected_features"]
    imputation_values = model_info["imputation_values"]

    X = pd.DataFrame(index=features_df.index)
    for feat in selected_features:
        if feat in features_df.columns:
            X[feat] = features_df[feat].copy()
        elif feat in imputation_values:
            X[feat] = imputation_values[feat]
        else:
            X[feat] = 0
    for col, val in imputation_values.items():
        if col in X.columns:
            X[col] = X[col].fillna(val)
    X = X.fillna(0)
    return X[selected_features]


def _ensemble_signals(
    models: list, features_df: pd.DataFrame, vote_threshold: float
) -> pd.DataFrame:
    """Vote + average-regressor over the ensemble.

    Returns a frame aligned to ``features_df.index`` with ``buy_signal`` (0/1),
    ``vote_fraction``, and ``conviction`` (mean regressor output across models,
    0 for non-buy rows). Mirrors EnsemblePredictor's vote/average logic.
    """
    all_votes = []
    all_returns = []
    for model_info in models:
        classifier = model_info["classifier"]
        regressor = model_info["regressor"]
        X = _build_model_matrix(features_df, model_info)
        votes = np.asarray(classifier.predict(X))
        all_votes.append(votes)
        if regressor is not None:
            preds = np.zeros(len(X))
            buy_mask = votes == 1
            if buy_mask.any():
                preds[buy_mask] = regressor.predict(X.loc[buy_mask])
            all_returns.append(preds)

    if not all_votes:
        return pd.DataFrame(
            {
                "buy_signal": np.zeros(len(features_df), dtype=int),
                "vote_fraction": np.zeros(len(features_df)),
                "conviction": np.zeros(len(features_df)),
            },
            index=features_df.index,
        )

    vote_fractions = np.asarray(all_votes, dtype=float).mean(axis=0)
    buy_signals = (vote_fractions >= vote_threshold).astype(int)
    conviction = (
        np.asarray(all_returns, dtype=float).mean(axis=0)
        if all_returns
        else np.zeros(len(features_df))
    )
    return pd.DataFrame(
        {
            "buy_signal": buy_signals,
            "vote_fraction": vote_fractions,
            "conviction": conviction,
        },
        index=features_df.index,
    )


def _attach_entry_costs(
    positions: pd.DataFrame, test_spreads: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Add a round-trip ``entry_cost`` = corwin_schultz_spread per position.

    The round-trip drag (enter near the ask, exit near the bid) equals the full
    quoted spread. Merges ``test_spreads`` (Ticker, Filing Date,
    corwin_schultz_spread) onto the positions (keyed Ticker + entry_date).
    Missing -> ``DEFAULT_ROUND_TRIP_COST``.
    """
    positions = positions.copy()
    if test_spreads is not None and not test_spreads.empty:
        spreads = test_spreads[
            ["Ticker", "Filing Date", "corwin_schultz_spread"]
        ].copy()
        spreads["Filing Date"] = pd.to_datetime(spreads["Filing Date"])
        merged = positions.merge(
            spreads,
            left_on=["Ticker", "entry_date"],
            right_on=["Ticker", "Filing Date"],
            how="left",
        )
        # Round-trip drag == full quoted spread (half-spread each on entry+exit).
        round_trip = merged["corwin_schultz_spread"]
        positions["entry_cost"] = round_trip.fillna(DEFAULT_ROUND_TRIP_COST).to_numpy()
    else:
        positions["entry_cost"] = DEFAULT_ROUND_TRIP_COST
    return positions


def backtest_strategy(
    strategy: tuple,
    model_type: str = "LightGBM",
    models_base_path: Optional[Path] = None,
    test_features_path: Optional[Path] = None,
    test_spreads_path: Optional[Path] = None,
    ohlcv_loader: Callable[..., pd.DataFrame] = _default_ohlcv_loader,
    spx_arrays: Optional[tuple[np.ndarray, np.ndarray]] = None,
    folds: Optional[list] = None,
    seeds: Optional[list] = None,
    vote_threshold: Optional[float] = None,
    test_features_df: Optional[pd.DataFrame] = None,
    test_spreads_df: Optional[pd.DataFrame] = None,
    model_loader: Callable[[Path], tuple] = _joblib_dir_loader,
    db_path: Optional[str] = None,
    max_spread_cost: Optional[float] = None,
    per_name_cap: Optional[float] = None,
    max_gross_exposure: float = 1.0,
) -> dict:
    """Backtest one ensemble strategy on the held-out test set.

    Parameters
    ----------
    strategy : (timepoint, tp, sl) tuple.
    model_type : label only (selects which xlsx ``run_all`` writes).
    models_base_path : root of ``{strategy_str}/fold_{f}/seed_{s}/``.
    test_features_path / test_spreads_path : parquet inputs (read if the
        corresponding ``*_df`` is not supplied directly — tests inject frames).
    ohlcv_loader, spx_arrays : injected price source + benchmark (no disk in tests).
    folds, seeds, vote_threshold : default to config values.
    model_loader : maps a model dir -> (classifier, regressor, metadata);
        overridable so tests can stub models without joblib/pickles.
    max_spread_cost : tradability liquidity filter on the full quoted spread;
        defaults to ``config.MAX_SPREAD_COST`` (filter ON). Names whose quoted
        spread (== round-trip entry_cost) exceeds this are dropped, mirroring the
        live PositionSizer. Pass a large value (e.g. ``float("inf")``) to disable.
    per_name_cap, max_gross_exposure : capital-model caps threaded into
        ``simulate_daily_portfolio`` (default 5% per name, 100% gross).

    Returns a dict of metrics keyed by ``strategy_str`` (see ``run_all``).
    """
    folds = folds or config.ENSEMBLE_FOLDS
    seeds = seeds or config.ENSEMBLE_SEEDS
    if vote_threshold is None:
        vote_threshold = config.ENSEMBLE_VOTE_THRESHOLD
    if max_spread_cost is None:
        max_spread_cost = config.MAX_SPREAD_COST
    if per_name_cap is None:
        per_name_cap = config.MAX_POSITION_SIZE
    models_base_path = Path(models_base_path or config.MODELS_PATH)

    timepoint, tp, sl = strategy
    strat_str = strategy_string(strategy)

    # --- Load the test set ---
    if test_features_df is None:
        if test_features_path is None:
            raise ValueError("Provide test_features_path or test_features_df")
        test_features_df = pd.read_parquet(test_features_path)
    test_features_df = test_features_df.copy()
    if "Filing Date" in test_features_df.columns:
        test_features_df["Filing Date"] = pd.to_datetime(
            test_features_df["Filing Date"]
        )

    test_spreads = test_spreads_df
    if test_spreads is None and test_spreads_path is not None:
        sp = Path(test_spreads_path)
        if sp.exists():
            test_spreads = pd.read_parquet(sp)

    # --- Load ensemble and vote ---
    models = _load_ensemble(strategy, models_base_path, folds, seeds, model_loader)
    if not models:
        raise FileNotFoundError(
            f"No models loaded for {strat_str} under {models_base_path}"
        )

    signals = _ensemble_signals(models, test_features_df, vote_threshold)

    buys = signals[signals["buy_signal"] == 1]
    if buys.empty:
        return {
            "strategy_str": strat_str,
            "Timepoint": timepoint,
            "TP": tp,
            "SL": sl,
            "sharpe": np.nan,
            "raw_sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "cagr": np.nan,
            "ann_vol": np.nan,
            "total_alpha": np.nan,
            "n_trades": 0,
            "n_days": 0,
        }

    # --- Conviction -> per-name CAPITAL weight ---
    # Min-max into [0.25, 1.0] (same as training/live), then scale by the 5% max
    # position size so weights are absolute capital fractions of 1.25%-5%.
    weights = calculate_position_sizes(buys["conviction"]) * config.MAX_POSITION_SIZE

    positions = pd.DataFrame(
        {
            "Ticker": test_features_df.loc[buys.index, "Ticker"].to_numpy(),
            "entry_date": test_features_df.loc[buys.index, "Filing Date"].to_numpy(),
            "weight": weights.to_numpy(),
        }
    )
    positions = _attach_entry_costs(positions, test_spreads)

    # --- Tradability filter: skip names the live system would reject ---
    # entry_cost is now the round-trip cost == full quoted spread, so compare it
    # directly. Mirrors PositionSizer dropping names above config.MAX_SPREAD_COST.
    if max_spread_cost is not None:
        keep = positions["entry_cost"] <= max_spread_cost
        positions = positions[keep].reset_index(drop=True)
        if positions.empty:
            return {
                "strategy_str": strat_str,
                "Timepoint": timepoint,
                "TP": tp,
                "SL": sl,
                "sharpe": np.nan,
                "raw_sharpe": np.nan,
                "sortino": np.nan,
                "max_drawdown": np.nan,
                "cagr": np.nan,
                "ann_vol": np.nan,
                "total_alpha": np.nan,
                "n_trades": 0,
                "n_days": 0,
            }

    # --- Daily MTM portfolio simulation ---
    horizon_days = horizon_business_days(timepoint)
    daily = simulate_daily_portfolio(
        positions,
        tp=tp,
        sl=sl,
        horizon_days=horizon_days,
        ohlcv_loader=ohlcv_loader,
        spx_arrays=spx_arrays,
        db_path=db_path,
        per_name_cap=per_name_cap,
        max_gross_exposure=max_gross_exposure,
    )

    alpha_metrics = compute_portfolio_metrics(
        daily["port_alpha"] if "port_alpha" in daily else pd.Series(dtype=float)
    )
    raw_metrics = compute_portfolio_metrics(
        daily["port_raw"] if "port_raw" in daily else pd.Series(dtype=float)
    )

    if "port_alpha" in daily and not daily["port_alpha"].dropna().empty:
        total_alpha = float((1.0 + daily["port_alpha"].fillna(0.0)).prod() - 1.0)
    else:
        total_alpha = np.nan

    return {
        "strategy_str": strat_str,
        "Timepoint": timepoint,
        "TP": tp,
        "SL": sl,
        "sharpe": alpha_metrics["sharpe"],
        "raw_sharpe": raw_metrics["sharpe"],
        "sortino": alpha_metrics["sortino"],
        "max_drawdown": alpha_metrics["max_drawdown"],
        "cagr": alpha_metrics["cagr"],
        "ann_vol": alpha_metrics["ann_vol"],
        "total_alpha": total_alpha,
        "n_trades": int(len(positions)),
        "n_days": alpha_metrics["n_days"],
    }


def run_all(
    strategies: Optional[list] = None,
    model_type: str = "LightGBM",
    models_base_path: Optional[Path] = None,
    test_features_path: Optional[Path] = None,
    test_spreads_path: Optional[Path] = None,
    ohlcv_loader: Callable[..., pd.DataFrame] = _default_ohlcv_loader,
    spx_arrays: Optional[tuple[np.ndarray, np.ndarray]] = None,
    out_dir: Optional[Path] = None,
    write_xlsx: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Backtest every strategy and (optionally) write the metrics workbook.

    Returns a DataFrame (one row per strategy). Strategies whose models are
    absent are skipped with a warning rather than aborting the whole run.
    """
    strategies = strategies if strategies is not None else config.STRATEGY_GRID
    rows = []
    for strategy in strategies:
        try:
            rows.append(
                backtest_strategy(
                    strategy,
                    model_type=model_type,
                    models_base_path=models_base_path,
                    test_features_path=test_features_path,
                    test_spreads_path=test_spreads_path,
                    ohlcv_loader=ohlcv_loader,
                    spx_arrays=spx_arrays,
                    **kwargs,
                )
            )
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {strategy}: {e}")
            continue

    report = pd.DataFrame(rows)

    if write_xlsx and not report.empty:
        out_dir = (
            Path(out_dir) if out_dir is not None else (config.ROOT_DIR / "results")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_type}_Backtest_Metrics.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            report.to_excel(writer, sheet_name="Backtest Metrics", index=False)
        print(f"[INFO] Wrote {out_path}")

    return report
