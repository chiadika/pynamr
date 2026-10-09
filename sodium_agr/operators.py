"""Linear operators for dual-echo AGR with T2* decay modelling.

* :class:`ApodizedNUFFT` – the "vstack of A_i" operator of the paper: the k-space
  samples are split into time bins and the image is multiplied by
  ``r ** tau_i`` before the (SENSE-)NUFFT of bin i, where ``r`` is the decay
  ratio image and ``tau_i`` the bin's readout time divided by the echo spacing.
* :func:`projected_gradient_operator` – the directional-TV projected gradient
  (Ehrhardt & Betcke 2016), the anatomical prior of the AGR.
* :class:`BowsherGradient` – nearest-neighbour (Bowsher) gradient, kept for
  the Bowsher-prior experiments.

Changes versus upstream are listed in ``REVIEW_JOURNAL.md`` (operators.py items):
numpy/cupy agnostic, cached decay powers, merged Bowsher forward/adjoint.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import sigpy
from numba import njit, prange

from .backend import array_module, xp

ndarray = "numpy.ndarray | cupy.ndarray"


class ApodizedNUFFT(sigpy.linop.Linop):
    """Time-binned (SENSE-)NUFFT with mono-exponential decay modelling.

    ``A x = vstack_i( F_i ( r**tau_i * x ) )``

    Parameters
    ----------
    Flist : list of sigpy Linops
        Fourier (or SENSE) operator of each time bin, all with the same
        ``ishape`` and an ``oshape`` of ``(num_coils, num_points_i, num_readouts)``.
    r : array
        real decay-ratio image, same shape as ``ishape``; values in (0, 1].
    tau : sequence of float
        readout time of each bin divided by the echo spacing.

    Notes
    -----
    The powers ``r ** tau_i`` and ``r ** (tau_i - 1)`` are computed once per
    assignment of ``r`` (property setter) instead of on every apply/adjoint,
    which upstream did ``2 * num_time_bins`` times per PDHG iteration.
    """

    def __init__(self, Flist: list, r, tau: Sequence[float]) -> None:
        ishape = tuple(Flist[0].ishape)
        oshape = (Flist[0].oshape[0], sum(int(F.oshape[1]) for F in Flist), Flist[0].oshape[2])
        super().__init__(oshape, ishape)
        self._Flist = list(Flist)
        self._tau = [float(t) for t in tau]
        if len(self._tau) != len(self._Flist):
            raise ValueError("one tau per time bin is required")
        self._xp = array_module(r)
        self._split = []
        offset = 0
        for F in self._Flist:
            n = int(F.oshape[1])
            self._split.append(slice(offset, offset + n))
            offset += n
        self._r = None
        self._powers = None
        self._powers_minus_one = None
        self.r = r

    # ---- properties -------------------------------------------------------
    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value) -> None:
        self._r = value
        self._xp = array_module(value)
        self._powers = [value ** t for t in self._tau]
        self._powers_minus_one = None  # computed lazily (only the ratio gradient needs it)

    @property
    def tau(self) -> list[float]:
        return self._tau

    @property
    def num_time_bins(self) -> int:
        return len(self._Flist)

    @property
    def xp(self):
        return self._xp

    def decay_power(self, i: int):
        """``r ** tau_i`` (cached)."""
        return self._powers[i]

    def decay_power_minus_one(self, i: int):
        """``r ** (tau_i - 1)`` (cached), used by the gradient with respect to r."""
        if self._powers_minus_one is None:
            self._powers_minus_one = [self._r ** (t - 1.0) for t in self._tau]
        return self._powers_minus_one[i]

    def bin_slice(self, i: int) -> slice:
        """slice of the point axis belonging to time bin i."""
        return self._split[i]

    def get_split_inds(self, i: int) -> np.ndarray:  # upstream name, kept
        s = self._split[i]
        return np.arange(s.start, s.stop)

    def F(self, i: int):
        """the Fourier/SENSE operator of bin i (without decay)."""
        return self._Flist[i]

    def get_Ai(self, i: int):  # upstream name, kept
        return self._Flist[i] * sigpy.linop.Multiply(self.ishape, self._powers[i])

    # ---- Linop interface --------------------------------------------------
    def _apply(self, input):
        y = self._xp.zeros(tuple(self.oshape), dtype=input.dtype)
        for i, F in enumerate(self._Flist):
            y[:, self._split[i], :] = F(self._powers[i] * input)
        return y

    def _adjoint(self, input):
        x = self._xp.zeros(tuple(self.ishape), dtype=input.dtype)
        for i, F in enumerate(self._Flist):
            x += self._powers[i] * F.H(input[:, self._split[i], :])
        return x

    def _adjoint_linop(self):
        return _ApodizedNUFFTAdjoint(self)


class _ApodizedNUFFTAdjoint(sigpy.linop.Linop):
    def __init__(self, A: ApodizedNUFFT) -> None:
        super().__init__(A.ishape, A.oshape)
        self._A = A

    def _apply(self, input):
        return self._A._adjoint(input)

    def _adjoint_linop(self):
        return self._A


# ---------------------------------------------------------------------------
def normalized_gradient_operator(ishape: Sequence[int]) -> sigpy.linop.Linop:
    """Finite-difference gradient scaled so that its operator norm is <= 1.

    The forward-difference operator with circular boundary has norm at most
    ``2 * sqrt(ndim)``; dividing by ``sqrt(4 * ndim)`` gives ``||G|| <= 1``.
    """
    ishape = tuple(int(n) for n in ishape)
    return (1.0 / math.sqrt(4 * len(ishape))) * sigpy.linop.FiniteDifference(ishape, axes=None)


def projected_gradient_operator(prior_image, eta: float = 0.0, backend=None) -> sigpy.linop.Linop:
    """Projected (directional-TV) gradient operator, https://doi.org/10.1137/15M1047325.

    ``PG x = G x - xi * sum_d( xi_d * (G x)_d )`` with ``xi`` the normalised
    gradient field of the prior image, so that gradients parallel to the prior's
    edges are removed from the penalty and gradients across edges absent in the
    prior are kept.

    Parameters
    ----------
    prior_image : array
        anatomical prior on the reconstruction grid (numpy or backend array).
        Real priors are the intended use; for a complex prior the real and
        imaginary parts of the gradient field are normalised separately, as
        upstream did, which is not the Hermitian projector onto the complement
        of ``span(xi)`` (see REVIEW_JOURNAL.md). Use a real prior.
    eta : float
        softening constant added in the normalisation, ``xi = g / sqrt(|g|^2 + eta^2)``.
    backend : module, optional
        numpy or cupy; default: the package backend.
    """
    mod = backend if backend is not None else xp
    prior = mod.asarray(prior_image)
    G = normalized_gradient_operator(prior.shape)
    xi = G(prior)
    real_norm = mod.sqrt(mod.linalg.norm(xi.real, axis=0) ** 2 + eta ** 2)
    if mod.iscomplexobj(xi):
        imag_norm = mod.sqrt(mod.linalg.norm(xi.imag, axis=0) ** 2 + eta ** 2)
        xi = xi.real / real_norm[None] + 1j * (xi.imag / imag_norm[None])
    else:
        xi = xi / real_norm[None]
    M = sigpy.linop.Multiply(G.oshape, xi)
    S = sigpy.linop.Sum(M.oshape, (0,))
    I = sigpy.linop.Identity(M.oshape)
    P = I - (M.H * S.H * S * M)
    return P * G


# ---------------------------------------------------------------------------
@njit(parallel=True)
def nearest_neighbors_3d(img, s, ninds):
    """n nearest (most similar) neighbours of every voxel inside a neighbourhood.

    img : 3D array; s : odd-sized binary neighbourhood; ninds : int output of
    shape (img.size, nnearest) receiving flattened C-order neighbour indices.
    """
    nnearest = ninds.shape[1]
    offsets = np.array(s.shape) // 2
    maxdiff = img.max() - img.min() + 1
    d12 = img.shape[1] * img.shape[2]
    d2 = img.shape[2]
    for i0 in prange(img.shape[0]):
        for i1 in range(img.shape[1]):
            for i2 in range(img.shape[2]):
                absdiff = np.zeros(s.shape)
                val = img[i0, i1, i2]
                i_flat = np.zeros(s.shape, dtype=ninds.dtype)
                for j0 in range(s.shape[0]):
                    for j1 in range(s.shape[1]):
                        for j2 in range(s.shape[2]):
                            t0 = i0 + j0 - offsets[0]
                            t1 = i1 + j1 - offsets[1]
                            t2 = i2 + j2 - offsets[2]
                            i_flat[j0, j1, j2] = t0 * d12 + t1 * d2 + t2
                            inside = (0 <= t0 < img.shape[0]) and (0 <= t1 < img.shape[1]) and (0 <= t2 < img.shape[2])
                            if inside and s[j0, j1, j2] == 1:
                                absdiff[j0, j1, j2] = np.abs(img[t0, t1, t2] - val)
                            else:
                                absdiff[j0, j1, j2] = maxdiff
                vox = i_flat[offsets[0], offsets[1], offsets[2]]
                ninds[vox, :] = i_flat.flatten()[np.argsort(absdiff.flatten())[:nnearest]]


class BowsherGradient(sigpy.linop.Linop):
    """Bowsher (nearest-neighbour) gradient: ``(B x)_i = x[nn_i] - x``.

    One class serves as forward and adjoint (``transpose=True``); upstream had
    two copies of the constructor differing only by a transpose.
    """

    def __init__(self, structural_image, neighborhood, num_nearest: int,
                 nearest_neighbor_inds=None, transpose: bool = False) -> None:
        self._structural_image = structural_image
        self._neighborhood = neighborhood
        self._num_nearest = int(num_nearest)
        self._transpose = bool(transpose)
        self._xp = array_module(structural_image)
        if self._xp is np:
            from scipy.sparse import csc_matrix
        else:
            from cupyx.scipy.sparse import csc_matrix
        self._csc_matrix = csc_matrix

        if nearest_neighbor_inds is None:
            inds = np.zeros((math.prod(structural_image.shape), self._num_nearest), dtype=np.int64)
            img_np = structural_image if self._xp is np else self._xp.asnumpy(structural_image)
            nb_np = neighborhood if isinstance(neighborhood, np.ndarray) else self._xp.asnumpy(neighborhood)
            nearest_neighbors_3d(np.asarray(img_np), np.asarray(nb_np), inds)
            self._nn = self._xp.asarray(inds)
        else:
            self._nn = nearest_neighbor_inds

        nvox = int(structural_image.size)
        rows = self._xp.arange(nvox, dtype=self._xp.int64)
        diag = self._csc_matrix((self._xp.full(nvox, -1.0), (rows, rows)), shape=(nvox, nvox))
        self._mats = []
        for i in range(self._num_nearest):
            off = self._csc_matrix((self._xp.full(nvox, 1.0), (rows, self._nn[:, i])), shape=(nvox, nvox))
            if self._transpose:
                off = off.T
            self._mats.append(diag + off)

        vol_shape = tuple(structural_image.shape)
        if self._transpose:
            super().__init__(oshape=vol_shape, ishape=(self._num_nearest,) + vol_shape)
        else:
            super().__init__(oshape=(self._num_nearest,) + vol_shape, ishape=vol_shape)

    @property
    def nearest_neighbor_inds(self):
        return self._nn

    def _apply(self, x):
        if self._transpose:
            out = self._xp.zeros(tuple(self.oshape), dtype=x.dtype)
            for i in range(self._num_nearest):
                out += self._xp.reshape(self._mats[i] @ x[i].ravel(), self.oshape)
            return out
        out = self._xp.zeros(tuple(self.oshape), dtype=x.dtype)
        for i in range(self._num_nearest):
            out[i] = self._xp.reshape(self._mats[i] @ x.ravel(), self.ishape)
        return out

    def _adjoint_linop(self):
        return BowsherGradient(self._structural_image, self._neighborhood, self._num_nearest,
                               nearest_neighbor_inds=self._nn, transpose=not self._transpose)


AdjointBowsherGradient = BowsherGradient  # upstream name
