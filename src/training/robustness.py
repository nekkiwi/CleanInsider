"""
Robustness / stress harness for the liquid insider strategy.

The liquid baseline (LightGBM, price>=$10 / ADV>=$5M, net target, 20 bps flat
round-trip cost) reports a net Sharpe ~4.72 and a large factor-neutral alpha on
the 2024-26 test set. Before trusting that number we stress every embedded
assumption:

  * ``sweep_entry_timing``  -- enter 0/1/2 trading days late (slippage in WHEN we
    actually fill, not just the price).
  * ``sweep_cost``          -- vary the flat round-trip cost (bps) and find the
    break-even where total alpha crosses 0.
  * ``sweep_liquidity``     -- tighten/loosen the price + ADV liquidity floors.
  * ``sweep_capital``       -- vary the per-name cap and the gross-exposure cap.
  * ``deflated_sharpe``     -- Bailey / Lopez de Prado deflated Sharpe: the
    probability the TRUE Sharpe > 0 after deflating for the number of strategy
    trials (multiple-testing / selection bias).
  * ``walk_forward_oos``    -- predict each fold's per-seed ensemble on THAT
    fold's genuine OOS validation window and pool the daily curves, spanning
    ~2013-2024 (incl. 2018 / 2020 / 2022 regimes).

Every function is dependency-injectable: it takes a ``backtest_fn`` (default the
real ``ensemble_backtest.backtest_strategy``) plus injected models / data /
loaders, so the unit tests run on synthetic inputs without touching disk or the
trained pickles. NO hard-coded paths.

Judgment calls (documented for the human running the real sweep)
----------------------------------------------------------------
* ``entry_offset`` semantics: offset=k shifts the WHOLE TP/SL/horizon window k
  trading days later (enter at Close[entry_idx+k]); offset=0 is today's
  behavior. See ``backtester.simulate_position_daily_alpha``.
* Cost override injection: in liquid mode ``backtest_strategy`` reads the flat
  cost from ``config.LIQUID_ROUND_TRIP_COST`` internally, so ``sweep_cost``
  temporarily sets that attribute around each call and restores it afterwards
  (``_override_config``). It never mutates config permanently. The same
  technique varies ``LIQUID_PRICE_MIN`` / ``LIQUID_ADV_MIN`` in
  ``sweep_liquidity``.
* Deflated Sharpe: computed from the chosen run's daily-return series (observed
  non-annualized SR, its skew/kurtosis) plus the trial count, following Bailey &
  Lopez de Prado (2014). ``deflated_sharpe`` returns Prob(true SR > 0).
"""

from __future__ import annotations

import contextlib
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from src import config
from src.training.backtester import (
    TRADING_DAYS,
    compute_portfolio_metrics,
    simulate_daily_portfolio,
)
from src.training.ensemble_backtest import (
    _ensemble_signals,
    _load_ensemble,
    backtest_strategy,
)
from src.training.training_helpers import (
    calculate_position_sizes,
    horizon_business_days,
)


# --------------------------------------------------------------------------- #
# Config override helper
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _override_config(**overrides):
    """Temporarily set ``src.config`` attributes, restoring them on exit.

    ``backtest_strategy`` reads several knobs (LIQUID_ROUND_TRIP_COST,
    LIQUID_PRICE_MIN, LIQUID_ADV_MIN) straight off ``config`` at call time, so a
    sweep that wants to vary them does so by patching the module attribute for
    the duration of the call instead of threading a new parameter through every
    layer. The original values are always restored, even on exception.
    """
    sentinel = object()
    saved = {k: getattr(config, k, sentinel) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(config, k, v)
        yield
    finally:
        for k, old in saved.items():
            if old is sentinel:
                with contextlib.suppress(AttributeError):
                    delattr(config, k)
            else:
                setattr(config, k, old)


def _metrics_row(result: dict, **extra) -> dict:
    """Pick the headline metrics off a ``backtest_strategy`` result dict."""
    row = {
        "sharpe": result.get("sharpe", np.nan),
        "raw_sharpe": result.get("raw_sharpe", np.nan),
        "sortino": result.get("sortino", np.nan),
        "max_drawdown": result.get("max_drawdown", np.nan),
        "cagr": result.get("cagr", np.nan),
        "ann_vol": result.get("ann_vol", np.nan),
        "total_alpha": result.get("total_alpha", np.nan),
        "n_trades": result.get("n_trades", 0),
        "n_days": result.get("n_days", 0),
    }
    row.update(extra)
    return row


# --------------------------------------------------------------------------- #
# 1. Entry-timing sweep
# --------------------------------------------------------------------------- #
def sweep_entry_timing(
    strategy: tuple,
    offsets: Sequence[int] = (0, 1, 2),
    backtest_fn: Callable[..., dict] = backtest_strategy,
    **bt_kwargs,
) -> pd.DataFrame:
    """Backtest ``strategy`` entering 0..k trading days late.

    One row per offset. offset=0 reproduces the baseline; positive offsets delay
    every fill, eroding any edge that decays in the first days after the filing.
    """
    rows = []
    for k in offsets:
        result = backtest_fn(strategy, entry_offset=int(k), **bt_kwargs)
        rows.append(_metrics_row(result, entry_offset=int(k)))
    cols = [
        "entry_offset",
        "sharpe",
        "raw_sharpe",
        "max_drawdown",
        "total_alpha",
        "cagr",
        "ann_vol",
        "sortino",
        "n_trades",
        "n_days",
    ]
    return pd.DataFrame(rows)[cols]


# --------------------------------------------------------------------------- #
# 2. Cost sweep (+ break-even)
# --------------------------------------------------------------------------- #
def sweep_cost(
    strategy: tuple,
    cost_bps: Sequence[float] = (0, 10, 20, 40, 60, 100),
    backtest_fn: Callable[..., dict] = backtest_strategy,
    liquid_mode: bool = True,
    **bt_kwargs,
) -> pd.DataFrame:
    """Vary the flat round-trip cost (in bps) charged per liquid position.

    For each cost the flat ``config.LIQUID_ROUND_TRIP_COST`` is overridden for
    the duration of the call (see ``_override_config``). Reports net Sharpe,
    total alpha and n_trades, plus a ``break_even`` flag marking the FIRST cost
    (in ascending order) at which ``total_alpha`` crosses from > 0 to <= 0 —
    i.e. the cost that wipes out the edge.
    """
    bt_kwargs.pop("liquid_mode", None)  # cost sweep is a liquid-mode concept
    ordered = sorted(float(c) for c in cost_bps)
    rows = []
    for bps in ordered:
        cost = bps / 10_000.0
        with _override_config(LIQUID_ROUND_TRIP_COST=cost):
            result = backtest_fn(strategy, liquid_mode=liquid_mode, **bt_kwargs)
        rows.append(_metrics_row(result, cost_bps=bps, cost=cost))

    df = pd.DataFrame(rows)
    # Break-even: first ascending cost where total_alpha is no longer positive.
    df["break_even"] = False
    alpha = pd.to_numeric(df["total_alpha"], errors="coerce")
    crossed = alpha.notna() & (alpha <= 0)
    if crossed.any():
        df.loc[crossed.idxmax(), "break_even"] = True
    cols = [
        "cost_bps",
        "cost",
        "sharpe",
        "total_alpha",
        "n_trades",
        "max_drawdown",
        "raw_sharpe",
        "n_days",
        "break_even",
    ]
    return df[cols]


# --------------------------------------------------------------------------- #
# 3. Liquidity-tier sweep
# --------------------------------------------------------------------------- #
def sweep_liquidity(
    strategy: tuple,
    tiers: Sequence[tuple] = ((10, 5e6), (20, 10e6), (50, 20e6), (5, 1e6)),
    backtest_fn: Callable[..., dict] = backtest_strategy,
    **bt_kwargs,
) -> pd.DataFrame:
    """Vary the liquid-universe floors (LIQUID_PRICE_MIN, LIQUID_ADV_MIN).

    One row per (price_min, adv_min) tier. Tighter floors shrink the tradeable
    set (fewer, more-liquid names); looser floors re-admit smaller caps. Always
    runs in liquid mode (the tiers are meaningless under the legacy CS filter).
    """
    bt_kwargs.pop("liquid_mode", None)
    rows = []
    for price_min, adv_min in tiers:
        with _override_config(
            LIQUID_PRICE_MIN=float(price_min), LIQUID_ADV_MIN=float(adv_min)
        ):
            result = backtest_fn(strategy, liquid_mode=True, **bt_kwargs)
        rows.append(
            _metrics_row(result, price_min=float(price_min), adv_min=float(adv_min))
        )
    cols = [
        "price_min",
        "adv_min",
        "sharpe",
        "total_alpha",
        "n_trades",
        "max_drawdown",
        "raw_sharpe",
        "cagr",
        "ann_vol",
        "n_days",
    ]
    return pd.DataFrame(rows)[cols]


# --------------------------------------------------------------------------- #
# 4. Capital-model sweep
# --------------------------------------------------------------------------- #
def sweep_capital(
    strategy: tuple,
    per_name: Sequence[float] = (0.025, 0.05, 0.10),
    gross: Sequence[float] = (0.5, 1.0, 2.0),
    backtest_fn: Callable[..., dict] = backtest_strategy,
    **bt_kwargs,
) -> pd.DataFrame:
    """Vary the per-name capital cap and the gross-exposure cap.

    Cartesian product of ``per_name`` x ``gross``. Larger per-name caps
    concentrate the book; larger gross caps add leverage (>1.0) or force more
    cash drag (<1.0). One row per (per_name_cap, max_gross_exposure).
    """
    bt_kwargs.pop("per_name_cap", None)
    bt_kwargs.pop("max_gross_exposure", None)
    rows = []
    for pn in per_name:
        for g in gross:
            result = backtest_fn(
                strategy,
                per_name_cap=float(pn),
                max_gross_exposure=float(g),
                **bt_kwargs,
            )
            rows.append(
                _metrics_row(
                    result, per_name_cap=float(pn), max_gross_exposure=float(g)
                )
            )
    cols = [
        "per_name_cap",
        "max_gross_exposure",
        "sharpe",
        "total_alpha",
        "max_drawdown",
        "cagr",
        "ann_vol",
        "n_trades",
        "n_days",
    ]
    return pd.DataFrame(rows)[cols]


# --------------------------------------------------------------------------- #
# 5. Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014)
# --------------------------------------------------------------------------- #
def _normal_cdf(x: float) -> float:
    """Standard-normal CDF via erf (no scipy dependency)."""
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _expected_max_sharpe(n_trials: int, var_trial_sr: float) -> float:
    """Expected maximum of ``n_trials`` i.i.d. N(0, var_trial_sr) Sharpes.

    Bailey/Lopez de Prado's order-statistic approximation:
        E[max] ~ sqrt(var) * [ (1-gamma) * Z^-1(1 - 1/N)
                               + gamma * Z^-1(1 - 1/(N*e)) ]
    with gamma the Euler-Mascheroni constant. This is the benchmark Sharpe the
    BEST of N random strategies would attain under the null of zero true edge.
    """
    from math import e

    n = max(int(n_trials), 1)
    if n == 1 or var_trial_sr <= 0:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    z1 = _inv_normal_cdf(1.0 - 1.0 / n)
    z2 = _inv_normal_cdf(1.0 - 1.0 / (n * e))
    return float(np.sqrt(var_trial_sr) * ((1.0 - gamma) * z1 + gamma * z2))


def _inv_normal_cdf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > p_high:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def deflated_sharpe(
    sharpes_or_returns,
    n_trials: int,
    var_trial_sr: Optional[float] = None,
) -> float:
    """Deflated Sharpe Ratio -> probability the TRUE Sharpe > 0.

    Parameters
    ----------
    sharpes_or_returns : the chosen run's DAILY RETURN series (preferred — the
        observed Sharpe, sample length, skew and kurtosis are derived from it),
        OR an array of per-trial annualized Sharpes (then the observed Sharpe is
        the max and the trial variance is the sample variance of the array).
    n_trials : number of strategy configurations tried (multiple-testing count).
        The more trials, the higher the benchmark a Sharpe must beat to be real.
    var_trial_sr : optional variance of the per-trial (non-annualized) Sharpes.
        Defaults: if an array of Sharpes is passed, the sample variance of that
        array (converted to per-observation units); if a return series is
        passed, ``1/sqrt(N)`` scale -> ``1/N`` variance under the null of i.i.d.
        N(0,1) trial Sharpes (a conservative standard assumption).

    Returns
    -------
    float in [0, 1] : the probability-statistic PSR/DSR — Prob(SR_true > 0)
        after deflating the observed Sharpe by the expected max under the null.
        Higher is better; ~0.95 is the usual significance bar.

    Method (Bailey & Lopez de Prado 2014)
    -------------------------------------
        DSR = Z( (SR_hat - SR_0) * sqrt(N-1)
                 / sqrt(1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2) )
    where SR_hat is the observed NON-annualized Sharpe over N observations,
    SR_0 = sqrt(var_trial_sr) * E[max of n_trials standard Normals], and skew /
    kurt are of the return series. With a flat (non-skewed, mesokurtic) series
    this reduces to the Gaussian PSR against the deflated benchmark.
    """
    arr = np.asarray(sharpes_or_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")

    # Heuristic: a "returns" series has many small values; a "sharpes" array is
    # a handful of O(1) annualized numbers. Treat <= n_trials short arrays whose
    # values are not tiny as a Sharpe array.
    looks_like_sharpes = (
        arr.size <= max(int(n_trials), 1) and np.nanmax(np.abs(arr)) > 0.5
    )

    if looks_like_sharpes:
        # Per-trial annualized Sharpes. Observed = max; benchmark from their var.
        sr_hat_ann = float(np.nanmax(arr))
        sr_hat = sr_hat_ann / np.sqrt(TRADING_DAYS)  # de-annualize to per-obs
        n_obs = max(arr.size, 2)
        skew = 0.0
        kurt = 3.0
        v_trial = (
            float(np.var(arr, ddof=1)) / TRADING_DAYS
            if var_trial_sr is None
            else float(var_trial_sr)
        )
    else:
        # Daily return series: derive everything from it.
        r = pd.Series(arr, dtype=float).dropna()
        n_obs = len(r)
        std = r.std(ddof=1)
        if not std or np.isnan(std):
            return float("nan")
        sr_hat = float(r.mean() / std)  # NON-annualized per-observation Sharpe
        skew = float(r.skew()) if n_obs > 2 else 0.0
        kurt = float(r.kurtosis() + 3.0) if n_obs > 3 else 3.0  # pandas excess->raw
        v_trial = 1.0 / n_obs if var_trial_sr is None else float(var_trial_sr)

    sr0 = _expected_max_sharpe(n_trials, v_trial)

    denom = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat**2
    if denom <= 0 or n_obs < 2:
        return float("nan")
    z = (sr_hat - sr0) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(_normal_cdf(z))


# --------------------------------------------------------------------------- #
# 6. Multi-regime walk-forward OOS
# --------------------------------------------------------------------------- #
def walk_forward_oos(
    strategy: tuple,
    models_base_path,
    features_dir,
    targets_dir,
    spx_arrays,
    ohlcv_loader: Callable[..., pd.DataFrame],
    folds: Optional[Sequence[int]] = None,
    seeds: Optional[Sequence[int]] = None,
    vote_threshold: Optional[float] = None,
    model_loader: Optional[Callable] = None,
    liquid_round_trip_cost: Optional[float] = None,
    price_min: Optional[float] = None,
    adv_min: Optional[float] = None,
    per_name_cap: Optional[float] = None,
    max_gross_exposure: float = 1.0,
    adv_loader: Optional[Callable[[int], pd.DataFrame]] = None,
    feature_filename: str = "validation_data.parquet",
) -> pd.DataFrame:
    """Genuine OOS walk-forward: each fold predicts on ITS OWN validation window.

    For each fold ``f`` in ``folds`` (default ``config.ENSEMBLE_FOLDS``):
      1. load that fold's per-seed models from
         ``models_base_path/{strategy_str}/fold_{f}/seed_{s}``;
      2. predict (vote + average regressor) on that fold's validation feature
         set (``features_dir/fold_{f}/{feature_filename}``);
      3. build liquid positions (price/ADV filter + flat round-trip cost);
      4. simulate the daily MTM portfolio and compute metrics.
    Each fold's validation window is genuinely OOS for that fold's models, so the
    pooled curve spans the full walk-forward history (~2013-2024 incl.
    2018/2020/2022). The last row (``fold == "OVERALL"``) is the metrics of the
    pooled daily curve (concatenation of every fold's port_alpha by date).

    Dependency-injected: ``ohlcv_loader``, ``spx_arrays``, ``model_loader`` and
    an optional ``adv_loader(fold)->DataFrame[Ticker, Filing Date, adv]`` keep
    the unit test synthetic. ``adv_min``/``price_min``/``liquid_round_trip_cost``
    default to the liquid config; ADV defaults to the per-fold ADV component if
    no ``adv_loader`` is supplied and the feature frame lacks an ``adv`` column.
    """
    from pathlib import Path

    folds = list(folds or config.ENSEMBLE_FOLDS)
    seeds = list(seeds or config.ENSEMBLE_SEEDS)
    if vote_threshold is None:
        vote_threshold = config.ENSEMBLE_VOTE_THRESHOLD
    if model_loader is None:
        from src.training.ensemble_backtest import _joblib_dir_loader

        model_loader = _joblib_dir_loader
    if liquid_round_trip_cost is None:
        liquid_round_trip_cost = getattr(config, "LIQUID_ROUND_TRIP_COST", 0.002)
    if price_min is None:
        price_min = getattr(config, "LIQUID_PRICE_MIN", 0.0)
    if adv_min is None:
        adv_min = getattr(config, "LIQUID_ADV_MIN", 0.0)
    if per_name_cap is None:
        per_name_cap = config.MAX_POSITION_SIZE

    timepoint, tp, sl = strategy
    horizon_days = horizon_business_days(timepoint)
    models_base_path = Path(models_base_path)
    features_dir = Path(features_dir)

    rows = []
    pooled_alpha: list[pd.Series] = []

    for fold in folds:
        # --- features for this fold ---
        feat_path = features_dir / f"fold_{fold}" / feature_filename
        if not feat_path.exists():
            print(f"[WARN] walk_forward_oos: missing {feat_path}, skipping fold {fold}")
            continue
        feats = pd.read_parquet(feat_path)
        feats = feats.copy()
        if "Filing Date" in feats.columns:
            feats["Filing Date"] = pd.to_datetime(feats["Filing Date"])

        # Merge ADV if absent (liquid filter needs Price + adv).
        if "adv" not in feats.columns:
            adv_df = None
            if adv_loader is not None:
                adv_df = adv_loader(fold)
            elif Path(getattr(config, "ADV_COMPONENT_PATH", "")).exists():
                adv_df = pd.read_parquet(config.ADV_COMPONENT_PATH)[
                    ["Ticker", "Filing Date", "adv"]
                ].drop_duplicates(["Ticker", "Filing Date"])
            if adv_df is not None and not adv_df.empty:
                adv_df = adv_df.copy()
                adv_df["Filing Date"] = pd.to_datetime(adv_df["Filing Date"])
                feats = feats.merge(adv_df, on=["Ticker", "Filing Date"], how="left")

        # --- this fold's ensemble (5 seeds) ---
        models = _load_ensemble(strategy, models_base_path, [fold], seeds, model_loader)
        if not models:
            print(f"[WARN] walk_forward_oos: no models for fold {fold}, skipping")
            continue

        daily, metrics = _simulate_fold(
            models=models,
            feats=feats,
            vote_threshold=vote_threshold,
            tp=tp,
            sl=sl,
            horizon_days=horizon_days,
            ohlcv_loader=ohlcv_loader,
            spx_arrays=spx_arrays,
            liquid_round_trip_cost=liquid_round_trip_cost,
            price_min=price_min,
            adv_min=adv_min,
            per_name_cap=per_name_cap,
            max_gross_exposure=max_gross_exposure,
        )
        rows.append({"fold": fold, **metrics})
        if daily is not None and "port_alpha" in daily and not daily.empty:
            pooled_alpha.append(daily["port_alpha"])

    # --- pooled OVERALL curve ---
    if pooled_alpha:
        pooled = pd.concat(pooled_alpha).groupby(level=0).sum().sort_index()
        overall = compute_portfolio_metrics(pooled)
        total_alpha = float((1.0 + pooled.fillna(0.0)).prod() - 1.0)
        rows.append(
            {
                "fold": "OVERALL",
                "sharpe": overall["sharpe"],
                "raw_sharpe": np.nan,
                "sortino": overall["sortino"],
                "max_drawdown": overall["max_drawdown"],
                "cagr": overall["cagr"],
                "ann_vol": overall["ann_vol"],
                "total_alpha": total_alpha,
                "n_trades": int(sum(r.get("n_trades", 0) for r in rows)),
                "n_days": overall["n_days"],
            }
        )

    cols = [
        "fold",
        "sharpe",
        "raw_sharpe",
        "sortino",
        "max_drawdown",
        "cagr",
        "ann_vol",
        "total_alpha",
        "n_trades",
        "n_days",
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[[c for c in cols if c in df.columns]]


def _simulate_fold(
    models,
    feats,
    vote_threshold,
    tp,
    sl,
    horizon_days,
    ohlcv_loader,
    spx_arrays,
    liquid_round_trip_cost,
    price_min,
    adv_min,
    per_name_cap,
    max_gross_exposure,
):
    """Vote -> liquid positions -> daily MTM -> metrics for ONE fold.

    Mirrors the liquid path of ``backtest_strategy`` but operates on an already
    loaded ensemble + feature frame (the OOS validation set). Returns
    (daily_df_or_None, metrics_dict).
    """
    nan_metrics = {
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

    signals = _ensemble_signals(models, feats, vote_threshold)
    buys = signals[signals["buy_signal"] == 1]
    if buys.empty:
        return None, nan_metrics

    weights = calculate_position_sizes(buys["conviction"]) * config.MAX_POSITION_SIZE
    positions = pd.DataFrame(
        {
            "Ticker": feats.loc[buys.index, "Ticker"].to_numpy(),
            "entry_date": feats.loc[buys.index, "Filing Date"].to_numpy(),
            "weight": weights.to_numpy(),
        }
    )

    # Liquid filter: Price + ADV from the feature frame.
    price = (
        pd.to_numeric(feats.loc[buys.index, "Price"], errors="coerce").to_numpy()
        if "Price" in feats.columns
        else np.full(len(positions), np.nan)
    )
    adv = (
        pd.to_numeric(feats.loc[buys.index, "adv"], errors="coerce").to_numpy()
        if "adv" in feats.columns
        else np.full(len(positions), np.nan)
    )
    positions["Price"] = price
    positions["adv"] = adv
    positions["entry_cost"] = float(liquid_round_trip_cost)

    keep = (pd.Series(price) >= price_min) & (pd.Series(adv) >= adv_min)
    positions = positions[keep.fillna(False).to_numpy()].reset_index(drop=True)
    if positions.empty:
        return None, nan_metrics

    daily = simulate_daily_portfolio(
        positions,
        tp=tp,
        sl=sl,
        horizon_days=horizon_days,
        ohlcv_loader=ohlcv_loader,
        spx_arrays=spx_arrays,
        per_name_cap=per_name_cap,
        max_gross_exposure=max_gross_exposure,
    )
    if daily is None or daily.empty or "port_alpha" not in daily:
        return None, {**nan_metrics, "n_trades": int(len(positions))}

    metrics = compute_portfolio_metrics(daily["port_alpha"])
    raw_metrics = compute_portfolio_metrics(daily["port_raw"])
    total_alpha = float((1.0 + daily["port_alpha"].fillna(0.0)).prod() - 1.0)
    return daily, {
        "sharpe": metrics["sharpe"],
        "raw_sharpe": raw_metrics["sharpe"],
        "sortino": metrics["sortino"],
        "max_drawdown": metrics["max_drawdown"],
        "cagr": metrics["cagr"],
        "ann_vol": metrics["ann_vol"],
        "total_alpha": total_alpha,
        "n_trades": int(len(positions)),
        "n_days": metrics["n_days"],
    }
