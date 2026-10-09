"""Array backend selection: cupy when a CUDA GPU is available, numpy otherwise.

Upstream imported ``cupy`` unconditionally, so nothing could even be imported on
a machine without CUDA. Everything in the package now goes through ``xp`` and
the helpers below; ``SODIUM_AGR_BACKEND=numpy`` forces the CPU path (useful for
tests and small runs, slow for real reconstructions).
"""
from __future__ import annotations

import os
from types import ModuleType
from typing import Any

import numpy as np

_FORCE = os.environ.get("SODIUM_AGR_BACKEND", "").strip().lower()

if _FORCE in ("", "cupy", "gpu"):
    try:
        import cupy as _cp  # type: ignore

        _cp.cuda.runtime.getDeviceCount()  # raises when no device is usable
        xp: ModuleType = _cp
        HAS_GPU = True
    except Exception:  # ImportError or CUDA runtime error
        if _FORCE in ("cupy", "gpu"):
            raise
        xp = np
        HAS_GPU = False
else:
    xp = np
    HAS_GPU = False

BACKEND_NAME = "cupy" if HAS_GPU else "numpy"


def to_device(array: Any, dtype=None):
    """Move an array to the compute backend (no copy if already there)."""
    out = xp.asarray(array)
    return out.astype(dtype, copy=False) if dtype is not None else out


def to_numpy(array: Any) -> np.ndarray:
    """Move an array back to the host as a numpy array."""
    if HAS_GPU:
        return xp.asnumpy(array)
    return np.asarray(array)


def array_module(array: Any) -> ModuleType:
    """numpy or cupy, whichever owns `array`."""
    if HAS_GPU:
        return xp.get_array_module(array)
    return np


def device_of(array: Any):
    """sigpy device object for `array` (CPU device for numpy arrays)."""
    import sigpy

    return sigpy.get_device(array)
