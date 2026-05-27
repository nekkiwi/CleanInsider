# tests/test_spread_estimator.py
"""Unit tests for the Corwin-Schultz spread estimator fix.

Guards against the regression where the `gamma` term was computed but never
assigned (dead code) and the beta term was erroneously squared, inflating
estimated spreads to a ~5% median.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scrapers.target_scraper.calculate_spread_estimator import (
    _calculate_corwin_schultz,
)


def _synthetic(n=80, half_range=0.01, drift=0.0008, seed=0):
    rng = np.random.default_rng(seed)
    price = 100 * np.cumprod(1 + rng.normal(drift, 0.01, n))
    return pd.DataFrame(
        {"High": price * (1 + half_range), "Low": price * (1 - half_range)}
    )


def test_zero_range_gives_zero_spread():
    # High == Low every day -> no range -> zero spread.
    df = pd.DataFrame({"High": [100.0] * 40, "Low": [100.0] * 40})
    spread = _calculate_corwin_schultz(df).dropna()
    assert not spread.empty
    assert np.allclose(spread.to_numpy(), 0.0)


def test_spread_finite_and_bounded():
    spread = _calculate_corwin_schultz(_synthetic()).dropna()
    assert not spread.empty
    assert np.isfinite(spread.to_numpy()).all()
    assert (spread >= 0).all()
    assert (spread <= 0.20).all()


def test_matches_hand_computed_corwin_schultz():
    # Constant 4%-range bars (H=102, L=98, flat price). For this input the
    # textbook Corwin-Schultz value is analytically:
    #   beta=2*ln(102/98)^2, gamma=ln(102/98)^2, den=3-2*sqrt(2)
    #   alpha=(sqrt(2*beta)-sqrt(beta))/den - sqrt(gamma/den)
    #   spread=2*(exp(sqrt(alpha))-1)/(1+exp(sqrt(alpha))) ~= 0.19932
    df = pd.DataFrame({"High": [102.0] * 40, "Low": [98.0] * 40})
    spread = _calculate_corwin_schultz(df, window=20).dropna()
    assert not spread.empty
    assert np.isclose(spread.iloc[-1], 0.19932, atol=1e-3)


def test_wider_ranges_increase_spread():
    narrow = _calculate_corwin_schultz(_synthetic(half_range=0.005)).dropna()
    wide = _calculate_corwin_schultz(_synthetic(half_range=0.03)).dropna()
    assert wide.mean() > narrow.mean()
