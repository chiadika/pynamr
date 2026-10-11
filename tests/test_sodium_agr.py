"""CPU tests for the sodium_agr package (run with ``SODIUM_AGR_BACKEND=numpy pytest tests``)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("SODIUM_AGR_BACKEND", "numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import sigpy  # noqa: E402

from sodium_agr import operators, preprocessing, reconstruction  # noqa: E402
from sodium_agr.backend import xp  # noqa: E402


def _radial_k(points=12, readouts=40, kmax=0.4):
    n = np.arange(1, readouts + 1)
    z = 1 - 2 * (n - 0.5) / readouts
    r = np.sqrt(np.maximum(0, 1 - z * z))
    az = np.deg2rad(137.508) * n
    d = np.stack([r * np.cos(az), r * np.sin(az), z], axis=1)
    radius = (np.arange(points) + 0.5) / points * kmax
    return (radius[:, None, None] * d[None, :, :]).astype(np.float32)       # (points, readouts, 3), 1/cm


def test_apodized_nufft_is_adjoint():
    rng = np.random.default_rng(0)
    N, fov, ncoils, bins = 8, 10.0, 2, 3
    k = _radial_k()
    sens = (rng.standard_normal((ncoils, N, N, N)) + 1j * rng.standard_normal((ncoils, N, N, N))).astype(np.complex64)
    r = rng.uniform(0.3, 1.0, (N, N, N)).astype(np.float32)
    Flist, tau = reconstruction.build_time_binned_operators(sens, k, fov, bins, sampling_time_us=5.0, TE_ms=0.08, delta_TE_ms=2.0)
    A = operators.ApodizedNUFFT(Flist, xp.asarray(r), tau)
    x = (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N))).astype(np.complex64)
    y = (rng.standard_normal(tuple(A.oshape)) + 1j * rng.standard_normal(tuple(A.oshape))).astype(np.complex64)
    lhs = np.vdot(y, A(x))
    rhs = np.vdot(A.H(y), x)
    assert abs(lhs - rhs) < 1e-3 * abs(lhs)
    # cached powers follow r
    A.r = xp.asarray(r ** 2)
    assert np.allclose(A.decay_power(1), (r ** 2) ** tau[1])


def test_projected_gradient_is_an_orthogonal_projection_for_real_priors():
    rng = np.random.default_rng(1)
    prior = rng.standard_normal((6, 6, 6))
    PG = operators.projected_gradient_operator(prior, eta=1e-3, backend=np)
    G = operators.normalized_gradient_operator(prior.shape)
    g = G(rng.standard_normal(prior.shape))
    xi = G(prior)
    xi = xi / np.sqrt((xi ** 2).sum(0))[None]                 # exact unit vectors -> exact projector
    P = lambda v: v - xi * (xi * v).sum(0)[None]
    assert np.allclose(P(P(g)), P(g), atol=1e-5)            # idempotent
    assert np.allclose(np.sum(xi * P(g), axis=0), 0, atol=1e-5)   # orthogonal to the prior gradient
    # the operator itself removes the prior's own gradient up to the eta softening
    residual = np.abs(PG(prior)).max() / np.abs(G(prior)).max()
    assert residual < 1e-2


def test_ratio_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    N, fov, ncoils, bins = 6, 10.0, 1, 2
    k = _radial_k(points=6, readouts=20)
    sens = np.ones((ncoils, N, N, N), np.complex64)
    F1, t1 = reconstruction.build_time_binned_operators(sens, k, fov, bins, 5.0, 0.08, 2.0)
    F2, t2 = reconstruction.build_time_binned_operators(sens, k, fov, bins, 5.0, 2.08, 2.0)
    r = rng.uniform(0.4, 0.9, (N, N, N)).astype(np.float64)
    A1, A2 = operators.ApodizedNUFFT(F1, r, t1), operators.ApodizedNUFFT(F2, r, t2)
    x = (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N))).astype(np.complex128)
    d1 = A1(x) * 0.7 + 0.01 * rng.standard_normal(tuple(A1.oshape))
    d2 = A2(x) * 0.7 + 0.01 * rng.standard_normal(tuple(A2.oshape))
    G = operators.normalized_gradient_operator((N, N, N))
    c0, g = reconstruction.r_cost_and_gradient(r.ravel(), x, A1, A2, d1, d2, G, beta_r=0.1)
    idx = rng.integers(0, r.size, 5); eps = 1e-5
    for i in idx:
        rp = r.ravel().copy(); rp[i] += eps
        c1, _ = reconstruction.r_cost_and_gradient(rp, x, A1, A2, d1, d2, G, beta_r=0.1)
        fd = (c1 - c0) / eps
        assert abs(fd - g[i]) < 1e-2 * max(1.0, abs(fd)), (fd, g[i])


def test_pdhg_step_sizes_satisfy_convergence_condition():
    sigma, tau = reconstruction.pdhg_step_sizes(0.1, 1.7)
    assert sigma * tau * 1.7 < 1
    # upstream's rule violated it for any stacked norm above 1 / 0.99
    assert 0.1 * (0.99 * 1.0 / 0.1) * 1.7 > 1


def test_preprocessing_writers_round_trip(tmp_path):
    k = _radial_k()
    params = preprocessing.TPIParameters(sampling_time_us=5.0, num_points=k.shape[0], num_cones=1,
                                         num_readouts_per_cone=[k.shape[1]], max_grad_G_cm=2.0, gamma_over_2pi_MHz_T=42.577)
    preprocessing.write_kspace_trajectory_h5(k, params, tmp_path / "kspace_trajectory.h5")
    k2, p2 = preprocessing.read_kspace_trajectory_h5(tmp_path / "kspace_trajectory.h5")
    assert np.allclose(k, k2) and p2 == params
    data = (np.random.default_rng(3).standard_normal((2, k.shape[0], k.shape[1])) * (1 + 1j)).astype(np.complex64)
    preprocessing.write_converted_data_h5(data, tmp_path / "raw_echo1" / "converted_data.h5")
    d2, rot = preprocessing.read_converted_data_h5(tmp_path / "raw_echo1" / "converted_data.h5")
    assert rot is None and np.allclose(data, d2)


def test_initial_ratio_handles_zero_denominator():
    from sodium_agr.pipeline import initial_ratio
    a1 = np.zeros((8, 8, 8)); a1[2:6, 2:6, 2:6] = 1.0; a1[3, 3, 3] = 0.0
    a2 = 0.5 * a1
    r0 = initial_ratio(a2, a1)
    assert np.isfinite(r0).all() and r0[3, 3, 3] == 1.0 and r0[4, 4, 4] == 0.5 and r0[0, 0, 0] == 1.0


# ---------------------------------------------------------------------------
# fast decay model (decay.py)
# ---------------------------------------------------------------------------
from sodium_agr import decay  # noqa: E402


def test_decay_basis_matches_exact_kernel():
    t = decay.sample_times_ms(312, 5.0, 0.08)
    basis = decay.DecayBasis(t, 2.19, (np.exp(-2.19 / 0.05), 1.0), tol=1e-3)
    rng = np.random.default_rng(4)
    r = np.exp(-2.19 / rng.uniform(0.05, 50.0, 2000))          # T2* from 0.05 to 50 ms
    psi = basis.psi_of_r(r)                                    # (K, 2000)
    approx = basis.c @ psi                                      # (312, 2000)
    exact = r[None, :] ** (t[:, None] / 2.19)
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert basis.num_segments <= 32 and rel < 5e-3, (basis.num_segments, rel)
    # derivative check
    _, dpsi = basis.psi_of_r(r, derivative=True)
    eps = 1e-6 * r
    fd = (basis.psi_of_r(r + eps) - basis.psi_of_r(r - eps)) / (2 * eps)
    good = (r > 1e-3)                                            # T2* > 0.32 ms; below, the log-grid derivative is ~5 % off
    rel = np.abs(dpsi[:, good] - fd[:, good]) / (np.abs(fd[:, good]) + 1e-3 * np.abs(fd[:, good]).max())
    assert np.median(rel) < 5e-3 and np.percentile(rel, 99) < 5e-2, (np.median(rel), np.percentile(rel, 99))


def test_segmented_decay_nufft_adjoint_and_gradient():
    rng = np.random.default_rng(5)
    N, fov, ncoils = 6, 10.0, 2
    k = _radial_k(points=8, readouts=24)
    sens = (rng.standard_normal((ncoils, N, N, N)) + 1j * rng.standard_normal((ncoils, N, N, N))).astype(np.complex64)
    r = rng.uniform(0.3, 1.0, (N, N, N)).astype(np.float64)
    A = reconstruction.build_decay_operators(sens, k, fov, 5.0, 0.08, 2.0, r, decay_model="segments", segment_tol=1e-4)
    x = (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N))).astype(np.complex128)
    y = (rng.standard_normal(tuple(A.oshape)) + 1j * rng.standard_normal(tuple(A.oshape))).astype(np.complex128)
    assert abs(np.vdot(y, A(x)) - np.vdot(A.H(y), x)) < 1e-6 * abs(np.vdot(y, A(x)))
    # agrees with the exact per-sample decay through the same Sense operator
    exact = np.zeros(tuple(A.oshape), np.complex128)
    for j in range(k.shape[0]):
        exact[:, j, :] = A.F(A.basis.exact_decay(r, j) * x)[:, j, :]
    assert np.linalg.norm(A(x) - exact) < 1e-3 * np.linalg.norm(exact)
    # ratio gradient by finite differences (segments) with the coarse/masked parametrisation
    A2 = reconstruction.build_decay_operators(sens, k, fov, 5.0, 2.08, 2.0, r, decay_model="segments", segment_tol=1e-4)
    d1 = A(x) * 0.8; d2 = A2(x) * 0.8
    G = operators.normalized_gradient_operator((N, N, N))
    mask = np.ones((N, N, N), bool); mask[0] = False
    param = decay.RatioParametrisation((N, N, N), mask=mask, factor=2, bounds=(0.01, 1.0))
    rho = param.initial_rho(r)
    c0, g = reconstruction.r_cost_and_gradient(rho.ravel(), x, A, A2, d1, d2, G, beta_r=0.1, param=param)
    for i in rng.integers(0, rho.size, 4):
        rp = rho.ravel().copy(); rp[i] += 1e-5
        c1, _ = reconstruction.r_cost_and_gradient(rp, x, A, A2, d1, d2, G, beta_r=0.1, param=param)
        fd = (c1 - c0) / 1e-5
        assert abs(fd - g[i]) < 2e-2 * max(1.0, abs(fd)), (fd, g[i])


def test_ratio_parametrisation_is_an_interpolation_with_exact_adjoint():
    rng = np.random.default_rng(6)
    param = decay.RatioParametrisation((9, 9, 9), factor=3, bounds=(0.0, 2.0))
    rho = np.full(param.coarse_shape, 0.7)
    r = param.to_fine(rho)
    assert np.allclose(r[2:-2, 2:-2, 2:-2], 0.7, atol=1e-6)      # partition of unity in the interior
    a = rng.standard_normal(param.coarse_shape); b = rng.standard_normal((9, 9, 9))
    assert abs(np.vdot(param.U(a), b) - np.vdot(a, param.U.H(b))) < 1e-9
