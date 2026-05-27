# file: src/training/model_factory.py
"""Estimator factory: map a ``model_type`` label to a (classifier, regressor) pair.

This is the single place where a model family's hyperparameters live, so the
training pipeline can stay family-agnostic. The DEPLOYED family, "LightGBM", is
preserved byte-for-byte (params identical to the historical ``_train_models``);
"TabPFN" is a research-only family gated behind CUDA.

TabPFN / torch are imported LAZILY inside ``make_estimators`` so that:
  * CPU CI (no tabpfn / torch / CUDA) never imports them, and
  * a CUDA-less environment fails with a clear RuntimeError rather than a
    confusing import error or a silent CPU run (TabPFN must never run on CPU here).
"""

from __future__ import annotations

# LightGBM is a hard project dependency, so importing it at module load is fine.
from lightgbm import LGBMClassifier, LGBMRegressor

# Historical LightGBM params from ModelTrainer._train_models. Kept verbatim so
# the deployed baseline + ensemble_backtest + parity tests stay identical.
_LIGHTGBM_PARAMS = {
    "n_jobs": -1,
    "verbosity": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def _make_lightgbm(seed: int):
    params = {"random_state": seed, **_LIGHTGBM_PARAMS}
    return LGBMClassifier(**params), LGBMRegressor(**params)


def _make_tabpfn(seed: int):
    """Build a CUDA-only TabPFN (classifier, regressor) pair.

    Raises RuntimeError if torch/tabpfn are unavailable or CUDA is not present,
    so this family can NEVER silently run on CPU (which would be intolerably
    slow and is not the intended deployment target).
    """
    # Import torch and gate on CUDA FIRST, before touching tabpfn. tabpfn's
    # import probes torch internals (e.g. torch.backends.mps), so importing it
    # with no/partial torch can raise unexpected errors; gating first keeps the
    # failure mode a clean RuntimeError on CPU.
    try:
        import torch
    except ImportError as exc:  # torch not installed (e.g. CPU CI)
        raise RuntimeError(
            "TabPFN model_type requested but torch could not be imported. "
            "TabPFN is research-only (see requirements-research.txt) and requires "
            "a CUDA GPU. Install with: pip install -r requirements-research.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "TabPFN model_type requested but CUDA is not available. TabPFN is "
            "GPU-only in this pipeline and must never run on CPU. Run on a CUDA "
            "machine (e.g. the GTX 1080 Ti) or use model_type='LightGBM'."
        )

    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except ImportError as exc:  # tabpfn not installed
        raise RuntimeError(
            "TabPFN model_type requested but tabpfn could not be imported. "
            "Install with: pip install -r requirements-research.txt"
        ) from exc

    common = {
        "device": "cuda",
        "n_estimators": 4,
        "ignore_pretraining_limits": True,
        "random_state": seed,
    }
    return TabPFNClassifier(**common), TabPFNRegressor(**common)


def make_estimators(model_type: str, seed: int) -> tuple:
    """Return ``(classifier, regressor)`` for the requested model family.

    Parameters
    ----------
    model_type : "LightGBM" (deployed, CPU-friendly) or "TabPFN" (research, CUDA-only).
    seed : random_state threaded into both estimators.

    Raises
    ------
    RuntimeError : ``model_type == "TabPFN"`` but torch/tabpfn/CUDA unavailable.
    ValueError   : unknown ``model_type``.
    """
    if model_type == "LightGBM":
        return _make_lightgbm(seed)
    if model_type == "TabPFN":
        return _make_tabpfn(seed)
    raise ValueError(
        f"Unknown model_type {model_type!r}. Expected 'LightGBM' or 'TabPFN'."
    )
