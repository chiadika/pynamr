"""Anatomically guided reconstruction (AGR) of dual-echo non-Cartesian MR data
with joint T2* decay estimation (Schramm et al., MRM 2024).

Modules: ``backend`` (numpy/cupy), ``operators``, ``reconstruction``,
``coils_sens``, ``registration``, ``preprocessing``, ``pipeline`` (the staged
reconstruction driven by :class:`pipeline.ReconConfig`).
"""
from .backend import BACKEND_NAME, HAS_GPU, xp  # noqa: F401

__all__ = ["BACKEND_NAME", "HAS_GPU", "xp"]
