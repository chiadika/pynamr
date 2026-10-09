"""Reconstruction algorithms: gridding, channel-wise least squares, regularised
SENSE (PDHG) and the alternating dual-echo AGR with decay estimation.

All functions accept numpy inputs and run on the package backend (GPU when
available). See ``REVIEW_JOURNAL.md`` (reconstruction.py items) for the changes
versus upstream; the important one is the PDHG step rule.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import sigpy
import sigpy.mri
from scipy.optimize import fmin_l_bfgs_b

from .backend import device_of, to_device, to_numpy, xp
from .operators import ApodizedNUFFT, normalized_gradient_operator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _coords(k, field_of_view_cm: float):
    """(num_points*num_readouts, 3) k-space coordinates in grid units on the backend."""
    return to_device(np.asarray(k, dtype=np.float32).reshape(-1, 3) * float(field_of_view_cm))


def operator_norm_squared(op: sigpy.linop.Linop, dtype, device, max_iter: int = 30) -> float:
    """Largest eigenvalue of ``op.H * op`` by power iteration."""
    return float(sigpy.app.MaxEig(op.H * op, dtype=dtype, device=device, max_iter=max_iter, show_pbar=False).run())


def pdhg_step_sizes(sigma: float, norm_sq_stacked: float, safety: float = 0.99) -> tuple[float, float]:
    """``(sigma, tau)`` satisfying ``sigma * tau * ||K||^2 = safety < 1``.

    Upstream used ``tau = 0.99 * ||S||^2 / sigma`` (and ``0.99 * sigma / 1.5``
    in the dual-echo code), which does not implement this condition.
    """
    return float(sigma), safety / (float(sigma) * float(norm_sq_stacked))


def _prox_and_cost(regularization: str, beta: float, oshape):
    if regularization == "L2":
        prox = sigpy.prox.L2Reg(oshape, lamda=beta)
        g = lambda z: float(beta * 0.5 * (z * z.conj()).sum().real)
    elif regularization == "L1":
        prox = sigpy.prox.L1Reg(oshape, lamda=beta)
        g = lambda z: float(beta * abs(z).sum())
    else:
        raise ValueError("regularization must be 'L1' or 'L2'")
    return prox, g


# ---------------------------------------------------------------------------
# simple reconstructions
# ---------------------------------------------------------------------------
def channelwise_ifft_recon(data: np.ndarray, k: np.ndarray, dcf: np.ndarray,
                           grid_shape=(64, 64, 64), field_of_view_cm: float = 22.0) -> np.ndarray:
    """Density-compensated gridding (adjoint NUFFT) per coil.

    data : (num_coils, num_points, num_readouts) complex; k : (num_points, num_readouts, 3)
    in 1/cm; dcf : (num_points, num_readouts). Returns (num_coils, *grid_shape).
    """
    coords = _coords(k, field_of_view_cm)
    w = to_device(np.asarray(dcf, dtype=np.float32).ravel())
    out = xp.zeros((data.shape[0],) + tuple(grid_shape), dtype=data.dtype)
    for i, d in enumerate(data):
        out[i] = sigpy.nufft_adjoint(to_device(d.ravel()) * w, coords, tuple(grid_shape))
    return to_numpy(out)


def channelwise_lsq_recon(data: np.ndarray, k: np.ndarray, grid_shape=(64, 64, 64),
                          field_of_view_cm: float = 22.0, max_iter: int = 10, **kwargs) -> np.ndarray:
    """Early-stopped conjugate-gradient least squares per coil (no regularisation)."""
    coords = _coords(k, field_of_view_cm)
    A = sigpy.linop.NUFFT(tuple(grid_shape), coords)
    out = xp.zeros((data.shape[0],) + tuple(grid_shape), dtype=data.dtype)
    for i, d in enumerate(data):
        out[i] = sigpy.app.LinearLeastSquares(A, to_device(d.ravel()), max_iter=max_iter,
                                              show_pbar=False, **kwargs).run()
    return to_numpy(out)


# ---------------------------------------------------------------------------
# regularised SENSE
# ---------------------------------------------------------------------------
def regularized_sense_recon(data: np.ndarray, coil_sens: np.ndarray, k: np.ndarray,
                            field_of_view_cm: float = 22.0, regularization: str = "L2",
                            beta: float = 1e-1, sigma: float = 1e-1,
                            operator_norm_squared: float | None = None, max_iter: int = 100,
                            G: sigpy.linop.Linop | None = None, u: np.ndarray | None = None,
                            legacy_step_sizes: bool = False, show_pbar: bool = False,
                            **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Regularised SENSE reconstruction with PDHG.

    minimises ``0.5 ||S x - d||^2 + beta * g(G x)`` with ``g`` the L1 or squared L2 norm.

    Parameters
    ----------
    data : (num_coils, num_points, num_readouts) complex
    coil_sens : (num_coils, nx, ny, nz) complex, scaled so that ``||S|| = 1``
    k : (num_points, num_readouts, 3) in 1/cm
    G : gradient-like operator for the prior; default: normalised finite differences
    u : initial dual variable (warm start)
    operator_norm_squared : ``||[S; G]||^2``; estimated by power iteration when None
    legacy_step_sizes : reproduce upstream's ``tau = 0.99 ||S||^2 / sigma``

    Returns
    -------
    (x, u) : reconstructed image and the final dual variable (both numpy)
    """
    coords = _coords(k, field_of_view_cm)
    d = to_device(data.reshape(data.shape[0], -1))
    S = sigpy.mri.linop.Sense(to_device(coil_sens), coord=coords)
    if G is None:
        G = normalized_gradient_operator(S.ishape)
    prox, g = _prox_and_cost(regularization, beta, G.oshape)

    if legacy_step_sizes:
        norm_s = 1.0 if operator_norm_squared is None else operator_norm_squared
        tau = 0.99 * norm_s / sigma
    else:
        if operator_norm_squared is None:
            K = sigpy.linop.Vstack([S, G])
            operator_norm_squared = operator_norm_squared_of(K, d.dtype, device_of(coords))
        sigma, tau = pdhg_step_sizes(sigma, operator_norm_squared)

    app = sigpy.app.LinearLeastSquares(S, d, G=G, proxg=prox, max_iter=max_iter, sigma=sigma,
                                       tau=tau, g=g, show_pbar=show_pbar, **kwargs)
    if u is not None:
        app.alg.u = to_device(u)
    x = app.run()
    return to_numpy(x), to_numpy(app.alg.u)


def operator_norm_squared_of(op: sigpy.linop.Linop, dtype, device, max_iter: int = 30) -> float:
    """``||op||^2`` by power iteration (small wrapper so callers need no sigpy details)."""
    return operator_norm_squared(op, dtype, device, max_iter=max_iter)


# ---------------------------------------------------------------------------
# dual-echo AGR with decay estimation
# ---------------------------------------------------------------------------
def build_time_binned_operators(coil_sens, k, field_of_view_cm: float, num_time_bins: int,
                                sampling_time_us: float, TE_ms: float, delta_TE_ms: float):
    """SENSE operators per time bin and their ``tau`` values for one echo.

    tau_i is the mean readout time of bin i (from excitation, TE included)
    divided by the echo spacing, so the decay of bin i is ``r ** tau_i``.
    """
    chunks = np.array_split(np.asarray(k, dtype=np.float32) * float(field_of_view_cm), num_time_bins, axis=0)
    sens_d = to_device(coil_sens)
    Flist = [sigpy.mri.linop.Sense(sens_d, to_device(c)) for c in chunks]
    t_ms = np.arange(k.shape[0]) * sampling_time_us * 1e-3 + TE_ms
    tau = [float(c.mean()) for c in np.array_split(t_ms / delta_TE_ms, num_time_bins)]
    return Flist, tau


def data_fidelity_gradient_r(x, A_1: ApodizedNUFFT, A_2: ApodizedNUFFT, d_1, d_2, residual_1=None, residual_2=None):
    """Gradient of the dual data fidelity with respect to the ratio image.

    ``d/dr 0.5||A_i(r) x - d_i||^2 = sum_bins tau_b r**(tau_b-1) Re( conj(x) F_b^H (F_b(r**tau_b x) - d_b) )``

    ``residual_i = A_i x - d_i`` may be passed to avoid recomputing the forward model.
    """
    mod = A_1.xp
    g_r = mod.zeros(tuple(A_1.ishape), dtype=A_1.r.dtype)
    xc = x.conj()
    for A, d, res in ((A_1, d_1, residual_1), (A_2, d_2, residual_2)):
        for i in range(A.num_time_bins):
            s = A.bin_slice(i)
            r_i = res[:, s, :] if res is not None else A.F(i)(A.decay_power(i) * x) - d[:, s, :]
            g_r += A.tau[i] * A.decay_power_minus_one(i) * (xc * A.F(i).H(r_i)).real
    return g_r


def r_cost_and_gradient(r_flat: np.ndarray, x, A_1: ApodizedNUFFT, A_2: ApodizedNUFFT, d_1, d_2,
                        G: sigpy.linop.Linop, beta_r: float) -> tuple[float, np.ndarray]:
    """Cost and gradient for the L-BFGS update of the ratio image (one forward pass).

    Upstream evaluated the forward model separately for the cost and for the
    gradient; here the residual is shared, which halves the cost of each L-BFGS step.
    """
    r = to_device(r_flat.reshape(tuple(A_1.ishape)), dtype=A_1.r.dtype)
    A_1.r = r
    A_2.r = r
    res_1 = A_1(x) - d_1
    res_2 = A_2(x) - d_2
    mod = A_1.xp
    data_fid = 0.5 * float((mod.abs(res_1) ** 2).sum() + (mod.abs(res_2) ** 2).sum())
    Gr = G(r)
    prior = 0.5 * beta_r * float((mod.abs(Gr) ** 2).sum())
    grad = data_fidelity_gradient_r(x, A_1, A_2, d_1, d_2, res_1, res_2) + beta_r * G.H(Gr).real
    return data_fid + prior, to_numpy(grad).astype(np.float64).ravel()


def dual_echo_sense_with_decay_estimation(
        data_1: np.ndarray, data_2: np.ndarray, sampling_time_us: float, TE1_ms: float, TE2_ms: float,
        coil_sens_1: np.ndarray, coil_sens_2: np.ndarray, k: np.ndarray, x0: np.ndarray, u0: np.ndarray,
        r0: np.ndarray, G: sigpy.linop.Linop, field_of_view_cm: float = 22.0, regularization: str = "L1",
        beta: float = 3e-3, sigma: float = 1e-1, max_iter: int = 100, max_outer_iter: int = 20,
        num_time_bins: int = 64, beta_r: float = 1e-3, r_bounds: tuple[float, float] = (0.01, 1.0),
        lbfgs_iter: int = 10, operator_norm_squared: float | None = None, legacy_step_sizes: bool = False,
        intermediate_dir: Path | str | None = None, callback: Callable | None = None, verbose: bool = True,
        **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Joint dual-echo AGR with mono-exponential decay estimation (alternating).

    Outer loop: (1) L-BFGS-B update of the ratio image ``r`` (bounded), (2) PDHG
    update of the image ``x`` with the decay-modelled stacked operator
    ``[A_1(r); A_2(r)]`` and the prior ``beta * ||G x||_1``.

    Notes
    -----
    * ``x`` is the magnetisation at t = 0 (TE is part of the modelled decay), not at TE1.
    * ``r = exp(-(TE2 - TE1) / T2*)``; ``r_bounds`` must allow the shortest T2* of interest.
    * ``intermediate_dir`` receives ``r_<n>.npy`` / ``x_<n>.npy`` per outer iteration;
      upstream wrote them to a temporary directory that was deleted on return.
    """
    d_1 = to_device(data_1)
    d_2 = to_device(data_2)
    x = to_device(x0, dtype=d_1.dtype)
    u = to_device(u0)
    r = to_device(r0, dtype=np.float32 if d_1.dtype == np.complex64 else np.float64)
    dTE = TE2_ms - TE1_ms

    Flist_1, tau_1 = build_time_binned_operators(coil_sens_1, k, field_of_view_cm, num_time_bins, sampling_time_us, TE1_ms, dTE)
    Flist_2, tau_2 = build_time_binned_operators(coil_sens_2, k, field_of_view_cm, num_time_bins, sampling_time_us, TE2_ms, dTE)
    A_1 = ApodizedNUFFT(Flist_1, r, tau_1)
    A_2 = ApodizedNUFFT(Flist_2, r, tau_2)
    prox, _ = _prox_and_cost(regularization, beta, G.oshape)

    if intermediate_dir is not None:
        intermediate_dir = Path(intermediate_dir)
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    bounds = int(r.size) * [tuple(r_bounds)]
    r_new = r
    for i_outer in range(max_outer_iter):
        if verbose:
            print(f"dual echo AGR with decay estimation, outer iteration {i_outer + 1:03d} / {max_outer_iter:03d}")
        res = fmin_l_bfgs_b(r_cost_and_gradient, to_numpy(A_1.r).astype(np.float64).ravel(),
                            args=(x, A_1, A_2, d_1, d_2, G, beta_r), maxiter=lbfgs_iter,
                            bounds=bounds, disp=1 if verbose else 0)
        r_new = to_device(res[0].reshape(tuple(A_1.ishape)), dtype=r.dtype)
        A_1.r = r_new
        A_2.r = r_new

        K = sigpy.linop.Vstack([A_1, A_2], axis=0)
        if legacy_step_sizes:
            tau = 0.99 * sigma / 1.5
            sig = sigma
        else:
            norm_sq = operator_norm_squared if operator_norm_squared is not None else \
                operator_norm_squared_of(sigpy.linop.Vstack([K, G]), d_1.dtype, device_of(d_1), max_iter=20)
            sig, tau = pdhg_step_sizes(sigma, norm_sq)
        app = sigpy.app.LinearLeastSquares(K, xp.concatenate((d_1, d_2), 0), G=G, proxg=prox, x=x,
                                           max_iter=max_iter, sigma=sig, tau=tau, show_pbar=False, **kwargs)
        app.alg.u = u
        x = app.run()
        u = app.alg.u.copy()

        if intermediate_dir is not None:
            np.save(intermediate_dir / f"r_{i_outer + 1}.npy", to_numpy(A_1.r))
            np.save(intermediate_dir / f"x_{i_outer + 1}.npy", to_numpy(x))
        if callback is not None:
            callback(i_outer, to_numpy(x), to_numpy(A_1.r))

    return to_numpy(x), to_numpy(r_new)
