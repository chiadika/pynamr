"""Faster decay modelling for the dual-echo AGR.

Two independent accelerations of the joint reconstruction with T2* decay:

1. :class:`DecayBasis` / :class:`SegmentedDecayNUFFT`: low-rank *time
   segmentation* of the decay kernel. The per-sample decay ``exp(-t R)`` with
   ``R = -ln(r) / dTE`` (``r`` the ratio image) is approximated by a truncated
   SVD ``exp(-t_j R) ~= sum_k c_k(t_j) psi_k(R)`` over the range of ``R`` the
   ratio bounds allow. The forward model then needs ``K`` NUFFTs over *all*
   samples instead of ``num_time_bins`` NUFFTs over sample subsets (upstream:
   64 bins, each with a full oversampled FFT). ``K`` is 6-16 for a 1e-3
   relative error, depending on the T2* range. This is the time-segmented
   NUFFT of Fessler & Sutton (2003) with the field map replaced by R2*.

2. :class:`RatioParametrisation`: the ratio image is optimised on a coarser
   grid and only inside a mask (``r = U rho`` with ``U`` trilinear upsampling,
   ``r = r_outside`` elsewhere). The ratio is smooth by construction (quadratic
   prior), so a factor 2-3 loses nothing while shrinking the L-BFGS-B problem
   by 8-27x and keeping the forward model unchanged.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.ndimage as ndimage
import sigpy

from .backend import array_module, to_device, to_numpy


# ---------------------------------------------------------------------------
# low-rank time segmentation
# ---------------------------------------------------------------------------
class DecayBasis:
    """Truncated SVD of ``exp(-t_j R)`` over the sample times ``t_j`` and a log grid of ``R``.

    Parameters
    ----------
    t_ms : (num_points,) sample times from excitation (TE included), ms
    delta_TE_ms : echo spacing, defines ``R = -ln(r) / delta_TE_ms`` (1/ms)
    r_bounds : (r_min, r_max) of the ratio image; sets the ``R`` range
    num_segments : number of SVD terms; 0 = smallest K with ``rel_error < tol``
    tol : relative Frobenius error of the truncated kernel (auto mode)
    num_R : size of the ``R`` grid
    max_segments : cap of the auto mode
    """

    def __init__(self, t_ms: Sequence[float], delta_TE_ms: float, r_bounds: tuple[float, float],
                 num_segments: int = 0, tol: float = 1e-3, num_R: int = 512, max_segments: int = 32) -> None:
        self.t_ms = np.asarray(t_ms, dtype=np.float64)
        self.delta_TE_ms = float(delta_TE_ms)
        r_min, r_max = float(r_bounds[0]), float(min(r_bounds[1], 1.0))
        R_max = -np.log(max(r_min, 1e-300)) / self.delta_TE_ms
        R_min = max(-np.log(r_max) / self.delta_TE_ms, 1e-6)
        self.R_grid = np.logspace(np.log10(R_min), np.log10(R_max), num_R)
        E = np.exp(-np.outer(self.t_ms, self.R_grid))                       # (num_points, num_R)
        U, s, Vt = np.linalg.svd(E, full_matrices=False)
        energy = np.cumsum(s ** 2)
        rel_err = np.sqrt(np.maximum(energy[-1] - energy, 0) / energy[-1])    # error after keeping k+1 terms
        if num_segments <= 0:
            K = int(np.searchsorted(-rel_err, -tol) + 1)                     # first k with rel_err <= tol
            K = int(min(max(K, 1), max_segments, len(s)))
        else:
            K = int(min(num_segments, len(s)))
        self.num_segments = K
        self.rel_error = float(rel_err[K - 1])
        self.c = (U[:, :K] * s[:K]).astype(np.float32)                        # (num_points, K) temporal weights
        self.psi = Vt[:K].astype(np.float64)                                 # (K, num_R) functions of R
        self.dpsi_dR = np.gradient(self.psi, self.R_grid, axis=1)
        self.r_min, self.r_max = r_min, r_max

    def R_of_r(self, r):
        xp = array_module(r)
        return -xp.log(xp.clip(r, self.r_min, self.r_max)) / self.delta_TE_ms

    def psi_of_r(self, r, derivative: bool = False):
        """``psi_k(R(r))`` for every k, shape (K, *r.shape); optionally also ``d psi_k / d r``."""
        xp = array_module(r)
        R = self.R_of_r(r)
        grid = xp.asarray(self.R_grid)
        psi = xp.stack([xp.interp(R, grid, xp.asarray(p)) for p in self.psi]).astype(r.dtype)
        if not derivative:
            return psi
        dR_dr = -1.0 / (xp.clip(r, self.r_min, self.r_max) * self.delta_TE_ms)
        dpsi = xp.stack([xp.interp(R, grid, xp.asarray(p)) for p in self.dpsi_dR]).astype(r.dtype) * dR_dr[None]
        return psi, dpsi

    def exact_decay(self, r, t_index: int):
        """``r ** (t_j / dTE)`` for a sample index (reference for tests)."""
        return r ** (self.t_ms[t_index] / self.delta_TE_ms)


class SegmentedDecayNUFFT(sigpy.linop.Linop):
    """``A x = sum_k C_k F( psi_k(r) * x )`` with ``F`` a (SENSE-)NUFFT over all samples and
    ``C_k`` the diagonal temporal weight ``c_k(t_j)`` along the point axis.

    Same interface as :class:`operators.ApodizedNUFFT` where the pipeline needs it
    (``r`` property, ``xp``, ``grad_r``).
    """

    def __init__(self, F: sigpy.linop.Linop, r, basis: DecayBasis) -> None:
        super().__init__(tuple(F.oshape), tuple(F.ishape))
        if int(F.oshape[1]) != basis.t_ms.size:
            raise ValueError(f"operator has {F.oshape[1]} points per readout, basis has {basis.t_ms.size}")
        self._F = F
        self.basis = basis
        self._xp = array_module(r)
        self._c = None
        self._r = None
        self._psi = None
        self._dpsi = None
        self.r = r

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value) -> None:
        self._r = value
        self._xp = array_module(value)
        self._psi = self.basis.psi_of_r(value)
        self._dpsi = None
        self._c = self._xp.asarray(self.basis.c)[None, :, None]              # broadcast over (coil, point, readout)

    @property
    def xp(self):
        return self._xp

    @property
    def num_segments(self) -> int:
        return self.basis.num_segments

    @property
    def F(self) -> sigpy.linop.Linop:
        return self._F

    def _apply(self, input):
        y = None
        for k in range(self.num_segments):
            yk = self._c[:, :, :, k] * self._F(self._psi[k] * input)
            y = yk if y is None else y + yk
        return y

    def _adjoint(self, input):
        x = None
        for k in range(self.num_segments):
            xk = self._psi[k] * self._F.H(self._c[:, :, :, k] * input)
            x = xk if x is None else x + xk
        return x

    def _adjoint_linop(self):
        return _SegmentedAdjoint(self)

    def grad_r(self, x, residual):
        """``d/dr 0.5 ||A(r) x - d||^2 = sum_k dpsi_k/dr * Re( conj(x) F^H (C_k residual) )``."""
        if self._dpsi is None:
            _, self._dpsi = self.basis.psi_of_r(self._r, derivative=True)
        xc = x.conj()
        g = None
        for k in range(self.num_segments):
            gk = self._dpsi[k] * (xc * self._F.H(self._c[:, :, :, k] * residual)).real
            g = gk if g is None else g + gk
        return g


class _SegmentedAdjoint(sigpy.linop.Linop):
    def __init__(self, A: SegmentedDecayNUFFT) -> None:
        super().__init__(A.ishape, A.oshape)
        self._A = A

    def _apply(self, input):
        return self._A._adjoint(input)

    def _adjoint_linop(self):
        return self._A


def sample_times_ms(num_points: int, sampling_time_us: float, TE_ms: float) -> np.ndarray:
    return np.arange(num_points) * sampling_time_us * 1e-3 + TE_ms


# ---------------------------------------------------------------------------
# coarse, masked ratio parametrisation
# ---------------------------------------------------------------------------
class RatioParametrisation:
    """``r = mask * (U rho) + (1 - mask) * r_outside`` with ``rho`` on a grid coarsened by ``factor``.

    ``U`` is trilinear interpolation (a sigpy ``Interpolate`` linop, whose adjoint
    is exact), so ``rho`` within the bounds keeps ``r`` within the bounds up to
    the array edge, where the result is clipped.
    """

    def __init__(self, fine_shape: Sequence[int], mask=None, factor: int = 2, r_outside: float = 1.0,
                 bounds: tuple[float, float] = (0.01, 1.0), backend=None) -> None:
        self.fine_shape = tuple(int(n) for n in fine_shape)
        self.factor = int(max(factor, 1))
        self.r_outside = float(r_outside)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.coarse_shape = tuple(int(np.ceil(n / self.factor)) for n in self.fine_shape)
        xp = backend if backend is not None else np
        self._xp = xp
        if self.factor == 1:
            self.U = sigpy.linop.Identity(self.fine_shape)
        else:
            grids = np.meshgrid(*[np.arange(n, dtype=np.float32) / self.factor - (nc // 2)
                                  for n, nc in zip(self.fine_shape, self.coarse_shape)], indexing="ij")
            coord = np.stack(grids, axis=-1)
            self.U = sigpy.linop.Interpolate(self.coarse_shape, to_device(coord), kernel="spline", width=2, param=1)
        self.mask = None if mask is None else to_device(np.asarray(mask, dtype=np.float32))

    @property
    def num_parameters(self) -> int:
        return int(np.prod(self.coarse_shape))

    def to_fine(self, rho):
        xp = array_module(rho)
        r = self.U(rho)
        r = xp.clip(r, self.bounds[0], self.bounds[1])
        if self.mask is not None:
            r = self.mask * r + (1.0 - self.mask) * self.r_outside
        return r

    def adjoint(self, grad_r):
        g = grad_r if self.mask is None else self.mask * grad_r
        return self.U.H(g)

    def initial_rho(self, r_fine: np.ndarray) -> np.ndarray:
        """coarse initial value from a fine ratio image (block mean on the coarse grid)."""
        r_fine = to_numpy(r_fine).astype(np.float64)
        if self.factor == 1:
            return r_fine
        z = ndimage.zoom(r_fine, [nc / n for nc, n in zip(self.coarse_shape, self.fine_shape)], order=1)
        return np.clip(z, self.bounds[0], self.bounds[1])


def object_mask(image, low_signal_fraction: float = 0.05, dilate: int = 2) -> np.ndarray:
    """largest connected object of ``|image| >= fraction * max``, dilated; used for the ratio mask."""
    a = np.abs(to_numpy(image))
    m = a >= low_signal_fraction * a.max()
    label, _ = ndimage.label(m)
    sizes = np.bincount(label.ravel())
    if sizes.size > 1:
        m = label == (sizes[1:].argmax() + 1)
    if dilate > 0:
        m = ndimage.binary_dilation(m, iterations=dilate)
    return m
