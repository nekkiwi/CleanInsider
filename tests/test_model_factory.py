"""
Unit tests for src/training/model_factory.make_estimators.

These tests must run on CPU CI with NO tabpfn/torch/CUDA available:
  * LightGBM path is exercised for real (lightgbm is a hard dependency).
  * TabPFN path is verified to FAIL CLEANLY (RuntimeError) when CUDA is
    unavailable or the tabpfn/torch import fails -- it must never attempt to
    build a TabPFN estimator in a CPU environment.
  * An unknown model_type raises ValueError.
"""

import builtins
import sys
from pathlib import Path

import pytest
from lightgbm import LGBMClassifier, LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.model_factory import make_estimators  # noqa: E402


def test_lightgbm_returns_lgbm_pair():
    clf, reg = make_estimators("LightGBM", 42)
    assert isinstance(clf, LGBMClassifier)
    assert isinstance(reg, LGBMRegressor)


def test_lightgbm_params_match_legacy():
    """The LightGBM params must be byte-for-byte the historical _train_models set."""
    clf, reg = make_estimators("LightGBM", 123)
    expected = {
        "random_state": 123,
        "n_jobs": -1,
        "verbosity": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    for est in (clf, reg):
        params = est.get_params()
        for key, val in expected.items():
            assert params[key] == val, f"{key}={params[key]!r} != {val!r}"


def test_unknown_model_type_raises_value_error():
    with pytest.raises(ValueError):
        make_estimators("XGBoost", 42)


def test_tabpfn_raises_runtimeerror_without_cuda(monkeypatch):
    """With torch present but CUDA off, TabPFN must raise a clear RuntimeError."""
    import types

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError):
        make_estimators("TabPFN", 42)


def test_tabpfn_raises_runtimeerror_when_torch_missing(monkeypatch):
    """If torch (or tabpfn) cannot be imported, TabPFN must raise RuntimeError,
    not ImportError/ModuleNotFoundError -- the CUDA-gate must be unambiguous."""
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch.") or name == "tabpfn":
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "tabpfn", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    with pytest.raises(RuntimeError):
        make_estimators("TabPFN", 42)
