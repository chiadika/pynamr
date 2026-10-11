# pynamr review journal

Fork of `gschramm/pynamr` (upstream `b32907d`), reviewed 2026-10-09 for the
7 T IR-UTE cones myelin project. Purpose: use the dual-echo AGR with decay
estimation (`sodium_agr/`) on proton UTE data, so the review is deepest there.
Each entry gives the file and line in upstream, what is wrong, how it was
fixed in this fork, and whether the fix changes numerical results.

Severity: **bug** = wrong result or crash; **ineff** = correct but wasteful;
**usability** = correct but hard to use or maintain; **flag** = behaviour that
may be intended, left unchanged, needs the owner's decision.

## Repository layout (what each folder is)

| folder | role | state |
| --- | --- | --- |
| `sodium_agr/` | the current real-data pipeline: NUFFT-SENSE PDHG recons, directional-TV AGR, joint decay (ratio) estimation | reviewed line by line, rewritten as a package |
| `sigpy/` | paper experiments on sigpy (brainweb simulations, GE/Siemens TPI, gradient studies); `utils_sigpy.py` is the older form of `sodium_agr/operators.py` | reviewed `utils_sigpy.py`, `gradient_test_utils.py`; the 20 experiment scripts were skimmed |
| `scripts/` | older simulation / GE recon scripts; most import a `pynamr` package and `pymirc` that are not in this repository | reviewed; legacy, several cannot run |
| `bowsher-cartesian-sodium-agr/` | Cartesian birdcage workflow with a Bowsher prior and L-BFGS on pseudo-complex arrays | reviewed |
| `tests/` | unit tests for the *old* `pynamr` package API | do not run against this repository (see below) |
| `wip/` | experiment | skimmed |

## sodium_agr (the pipeline we use)

### reconstruction.py

1. **bug** `regularized_sense_recon`, line 190: PDHG step `tau = 0.99 * operator_norm_squared / sigma`.
   The convergence condition of PDHG is `sigma * tau * ||K||^2 < 1` with K the
   *stacked* operator `[S; G]`, so tau must be `0.99 / (sigma * ||K||^2)`.
   The code divides the wrong way and uses `||S||^2 = 1` (the coil maps are
   scaled so that `||S|| = 1`), ignoring `||G||^2 <= 1`. With the defaults
   `sigma = 0.1`, `tau = 9.9` the product is `0.99 * ||K||^2`, up to 1.98 for
   `||K||^2 = 2`, which can diverge. It works in practice because the stacked
   norm is usually about 1.5 and PDHG tolerates mild violations for a few
   hundred iterations. The dual-echo function (line 478) uses the unrelated
   expression `tau = 0.99 * sigma / 1.5`, an empirical patch for the same
   problem.
   *Fix:* `tau = 0.99 / (sigma * norm_sq)` with `norm_sq` estimated by power
   iteration on `K^H K` (cached), or the conservative bound `||S||^2 + ||G||^2`.
   `legacy_step_sizes=True` reproduces the old values for comparison runs.
   *Effect:* the fixed point is unchanged; the iterates differ, so results at a
   finite iteration count differ slightly.
2. **bug** `regularized_sense_recon`, line 180: `sigpy.app.MaxEig(..., device=data.device)`;
   `data` is a numpy array and has no `.device`, so `operator_norm_squared=None`
   raises `AttributeError`. *Fix:* use `sigpy.get_device(k_d)`.
3. **bug** `dual_echo_sense_with_decay_estimation`, line 443: intermediate results
   are written into `tempfile.TemporaryDirectory()`, which is deleted when the
   function returns. The files can never be read; every outer iteration pays
   the I/O. *Fix:* `intermediate_dir: Path | None`, nothing is written when it is `None`.
4. **ineff** `r_cost_wrapper` / `r_gradient_wrapper`, lines 272-325: each L-BFGS
   evaluation copies the ratio image twice (`r_init = A_1.r.copy()` and the
   restore) and transfers `r_flat` host->device; the two wrappers also compute
   the forward model independently, so every L-BFGS step costs two full
   dual-echo forward passes. *Fix:* one `r_cost_and_gradient` callable used
   through `fmin_l_bfgs_b(..., fprime=None)` with `approx_grad=False`, sharing
   the residual between cost and gradient (halves the work of the r update).
5. **ineff** `data_fidelity_gradient_r`, line 237: `A.r ** (tau - 1)` recomputed
   for every time bin on every call. Fixed through the cached powers in the operator (operators.py item 2).
6. **usability** `regularized_sense_recon` returns `(x, u)` always because `G`
   is never `None` after line 167, contrary to the docstring. Return type made consistent and documented.
7. **flag** `dual_echo_sense_with_decay_estimation`, line 447: ratio bounds are
   `(0.01, 1)`; the master script initialises `r0` with `np.clip(.., 0, 1)` and
   sets 1 outside the main clump. A lower bound of 0.01 corresponds to
   `T2* >= dTE / ln(100)`; for proton UTE with dTE 2.2 ms that is 0.48 ms, which
   excludes the myelin pool entirely. Exposed as `r_bounds`, default unchanged.
8. **flag** the decay model `r ** (t / dTE)` with `t` counted from the
   excitation (TE included) means the reconstructed `x` is the magnetisation
   at t = 0, not at TE1. This is the paper's definition and is kept, but it is
   now stated in the docstring and the CLI help because it matters for anyone
   comparing `x` with a gridded TE1 image.

### operators.py

1. **bug** `ApodizedNUFFT._apply`, line 102: `y = cp.zeros(...)` while `_adjoint`
   uses `self._xp.zeros`; the operator cannot run with a numpy backend even
   though it carries `xp`. *Fix:* `self._xp.zeros`.
2. **ineff** `get_Ai`, line 97: builds `Multiply(ishape, r ** tau_i)` on every
   forward/adjoint call, i.e. `2 * num_time_bins` full-grid power operations per
   PDHG iteration (128 per iteration at 64 bins). *Fix:* powers are cached when
   `r` is set and reused; `r ** (tau_i - 1)` for the ratio gradient is cached too.
3. **usability** `projected_gradient_operator`: the real and imaginary parts of
   the prior gradient are normalised separately, and the projector is
   `I - M^H S^H S M`, i.e. `g - conj(xi) <xi, g>_unconjugated`. For a real prior
   image (every use in this repository) this is exactly the dTV projector of
   Ehrhardt & Betcke. For a complex prior it is not the projection onto the
   complement of `span(xi)`. Documented; behaviour unchanged; a test asserts
   idempotence and orthogonality for real priors.
4. **ineff** `projected_gradient_operator`, lines 174-180: `where(norm > 0)` is
   always all-true for `eta > 0`; the per-component loop with fancy indexing
   copies the field three times. Replaced by a single vectorised division.
5. **usability** `BowsherGradient` / `AdjointBowsherGradient`: 120 lines
   duplicated; the adjoint only differs by a transpose of the off-diagonal
   matrices. Merged into one class with a `transpose` flag; sparse index
   arrays built with integer dtype (the float `arange` on line 334 relies on
   implicit casting that `cupyx.scipy.sparse` does not do).
6. **usability** type hints reference an undefined `xp` (`structural_image: xp.ndarray`);
   harmless only because of `from __future__ import annotations`. Replaced by the module `ndarray` alias.

### sodium_dual_echo_agr.py (master script)

1. **bug** line 274: `iff_aff = data['ifft_aff']` (typo) when loading cached
   IFFTs; the affine is never used afterwards, so the only effect is a dead
   variable, but anything that later relied on `ifft_aff` would get a `NameError`.
2. **flag / probable bug** line 479: the AGR of the *second* echo is initialised
   with `x=cp.asarray(sense_L2_1)`, the L2 recon of the *first* echo, while the
   first echo's AGR uses `sense_L2_1` too. Almost certainly meant `sense_L2_2`.
   Only the starting point changes, so the converged result is the same; at
   500 iterations the difference is small but non-zero. Fixed (initialise with
   the matching echo); `--legacy_init` keeps the old behaviour.
3. **bug** line 501: `r0 = np.clip(np.abs(agr_L1_2) / np.abs(agr_L1_1), 0, 1)`
   divides by zero where the first echo is exactly 0 (outside the object after
   an L1 prior this does happen) and `np.clip` passes NaN through; the clump
   mask fixes the outside but not zeros inside the clump. *Fix:* `np.divide(..., where=denominator > 0)` with 1 elsewhere.
4. **bug** `channelwise_ifft_recon` is called with a hard-coded `grid_shape=(64, 64, 64)`
   regardless of `--matrix_size`; fine for 64 or 128 sodium, wrong for any
   other geometry (the IFFT is only used for a coil-combined preview, so no
   downstream effect). Now `min(matrix_size, 64)` cubed, documented.
5. **usability** `TE1_ms = 0.5`, `TE2_ms = 5.`, `eta = 5e-3`, `num_time_bins = 64`,
   `max_iter = 100` inside the decay loop, `smoothing = 7` for the coil maps,
   the data normalisation point (`data[:, 1, :]`), and the GE `rotation`
   handling are hard-coded. All are now CLI / config options with the old values as defaults.
6. **flag** line 414: `na_origin = anat_origin.copy()` assumes the sodium
   volume's world origin equals the anatomical NIfTI's origin before
   registration. That is a geometry hack that happens to work for the GE data
   layout. Replaced by an explicit `--sodium_origin_mm` (default: FOV centred at 0,
   which is what a centred reconstruction means) and kept the old behaviour
   behind `--origin_from_anat`.
7. **usability** the script is a 550-line straight-line program with stage
   caching done by hand (`if not file.exists(): ... else: load`). Rewritten
   as `sodium_agr/pipeline.py` with one function per stage, a `Stage` cache
   helper, a `ReconConfig` dataclass, and a thin CLI (`sodium_dual_echo_agr.py`
   keeps its name and options so existing commands still work).
8. **usability** `cupy` is imported unconditionally at module level, so the
   package cannot even be imported on a machine without CUDA. `sodium_agr/backend.py`
   selects cupy when available and numpy otherwise (`SODIUM_AGR_BACKEND=numpy`
   forces CPU). The CPU path is slow but makes tests and small runs possible.

### coils_sens.py

1. **bug** `smooth`, line 193: `ndimage.filters.uniform_filter` — the
   `scipy.ndimage.filters` namespace was deprecated in SciPy 1.8 and removed
   in 1.13; the pinned environment (`numpy<=1.19`, `sigpy==0.1.25`) is the only
   reason it still ran. *Fix:* `ndimage.uniform_filter`.
2. **bug** `calculate_csm_inati_iter`, line 150: the fallback in the `except`
   branch sums `im` instead of `D` (copy-paste from line 120). Unreachable on
   any numpy >= 1.7; fixed anyway and the bare `except` narrowed.
3. **ineff** the Inati loop runs on numpy even when the data live on the GPU
   (arrays are moved back and forth around it). Left on numpy on purpose, the
   smoothing is the cost and it is small at 64^3-128^3; documented.

### preprocessing.py

1. **bug** `convert_ome_data`, lines 352-359: the HDF5 file is only written
   when `output_file is None`; passing an explicit output path silently writes
   nothing and returns the array. *Fix:* write in both cases.
2. **usability** `read_tpi_akwav_kspace_trajectory`: `N["params"]` offset is
   recomputed from scratch (`256 + 4 + N["groups"] * 2`) instead of the running
   `offset`; equal by construction but fragile. Uses `offset`.
3. **usability** no writer for a trajectory that is not a TPI gradient file.
   Added `write_kspace_trajectory_h5(k, params, path)` and
   `write_converted_data_h5(data, path)` so other sequences (our cones) can
   produce the expected layout without hand-rolled h5py code.

### registration.py

1. **usability** `voxel_size.astype(...)` requires numpy arrays although the
   signature says `Sequence[float]`; `np.asarray` added. Identical function
   bodies exist in `scripts/utils.py` and `scripts/preprocess_ge_na_data.py`;
   the package version is now the single implementation.
2. **robustness** the fixed pyramid `[4, 2, 1]` and the fixed 10 % metric
   sampling make `ImageRegistrationMethod` abort with "All samples map outside
   moving image buffer" on small volumes (found with a 16^3 smoke test).
   Levels that would shrink the fixed image below 8 voxels are now dropped
   and at least ~5000 samples are drawn. Both rules are no-ops for the
   128^3-265^3 volumes the pipeline is meant for.
3. **robustness** the same ITK error also comes from the optimiser: with the
   upstream learning rate 0.2 (scales from physical shift) a coarse fixed image
   (32^3 reduced in-vivo run) is thrown entirely out of overlap at the first
   step. `align_images` now retries with the learning rate halved (up to 3
   times) and reports it.

### pipeline.py (new) and the thin CLI

The 550-line master script is split into `pipeline.ReconConfig` (every
constant that was hard-coded: TE1/TE2, eta, bins, r bounds, preview matrix,
origin policy) and `pipeline.run`, eight cached stages with a tiny `Stage`
helper replacing eleven copies of the `if file.exists(): load else: compute,
save` pattern. The CLI keeps all upstream options and adds
`--legacy_step_sizes`, `--legacy_init`, `--origin_from_anat`,
`--sodium_origin_mm`, `--TE1_ms/--TE2_ms`, `--eta`, `--num_time_bins`,
`--r_bounds`, `--save_intermediates`, `--output_subdir`. Stage files are npz
(image + dual variable together) instead of separate `.npy`; the NIfTI
outputs keep their upstream names. `initial_ratio` no longer divides by zero
and keeps the largest connected object only.

End-to-end check (CPU, numpy backend): a 16^3 two-coil two-compartment phantom
with decay ratios 0.5 / 0.9 reconstructs to median ratios 0.51 / 0.91 after
3 outer iterations.

### decay.py (new): faster decay modelling

Two accelerations of the joint stage, both exact to a tolerance and both
optional (`--decay_model segments`, `--r_coarse_factor`, `--r_mask`):

1. **Low-rank time segmentation.** Upstream models the in-readout decay with
   64 sample bins, each a full (SENSE-)NUFFT with its own oversampled FFT, so
   one operator application costs 64 FFTs per coil. `DecayBasis` takes the
   SVD of `exp(-t_j R)` over the sample times and a log grid of `R = -ln r /
   dTE` spanning the ratio bounds; `SegmentedDecayNUFFT` applies `sum_k C_k
   F(psi_k(r) x)`. For our readout (312 x 5 us, TE 80 us, dTE 2.19 ms) and
   T2* down to 0.05 ms, **5 terms** give a 4e-4 relative kernel error; with
   upstream's bounds (T2* >= 0.48 ms) 3 terms suffice. The ratio gradient is
   `sum_k dpsi_k/dr Re(conj(x) F^H C_k res)` with `dpsi/dR` tabulated on the grid
   (accurate to ~1e-3 for T2* > 0.3 ms, ~5 % below that, where the signal is
   gone anyway). Same mathematics as Fessler & Sutton's time-segmented NUFFT
   for field maps.
2. **Coarse, masked ratio.** `RatioParametrisation`: `r = mask * U rho + (1 -
   mask)`, `U` trilinear upsampling (sigpy `Interpolate`, exact adjoint), rho on
   a grid coarsened by `r_coarse_factor`; L-BFGS-B runs on rho with the same
   bounds. The quadratic prior on r makes it smooth, so factor 2 changes
   nothing visible while dividing the L-BFGS problem by 8 (and by the
   mask fraction on top).

Unit tests: kernel accuracy vs the exact decay, operator adjointness and
agreement with the exact per-sample decay, finite-difference gradient through
the parametrisation, interpolation partition of unity and adjointness.

**Measured (CPU, sigpy):** the segmentation reproduces the 64-bin result
(image NRMSE 0.000, ratio RMSE 0.001 on the reduced in-vivo subject) but is
*not* faster with sigpy's NUFFT, which is interpolation-bound: one
full-sample NUFFT at 128^3 costs 14.7 s of which the FFT is 0.6 s, so five
full-sample NUFFTs cost about what 64 partial ones do (forward 98 vs 162 s,
adjoint 87 vs 72 s at 128^3, 2 coils, 2.5 M points per coil; full toy run
1190 vs 378 s). Expect ~1.3x at best on a GPU. The speed-up that the
segmentation promises needs an interpolation-free (Toeplitz) normal operator
with a gradient-based x-update, which is a solver change. Keep
`--decay_model segments` as the accurate option, not the fast one.

### tests/

1. **bug** both test modules import `pynamr` (`pynamr.TPIReadOutTime`,
   `pynamr.MonoExpDualTESodiumAcqModel`, ...), an older package that is not in
   this repository, and `test_gradients.py` imports `IPython.core.debugger`.
   They cannot be collected. Replaced by `tests/test_sodium_agr.py`, which
   runs on CPU: adjointness of `ApodizedNUFFT`, idempotence/orthogonality of the
   projected gradient, finite-difference check of the ratio gradient, PDHG
   step-size sanity, and the preprocessing writers.

## sigpy/ (paper experiments)

1. **bug** `gradient_test_utils.py`, `cost_and_grad_wrapper`, lines 44-50:
   the cost uses `0.5 * ||G x||^2` but the gradient uses `0.5 * G^H G x`; the
   gradient of `0.5 ||Gx||^2` is `G^H G x`. The L-BFGS runs in
   `gradient_test*.py` therefore minimise an objective whose prior weight is
   half of what the printed cost says, and the line search sees inconsistent
   cost/gradient pairs. *Fix:* `grad_prior = G.H(G(x))`. Changes results of
   those experiment scripts (prior effectively doubled relative to before).
2. **ineff** `utils_sigpy.py`, `NUFFTT2starDualEchoModel`: `get_operators_w_decay_model`
   rebuilds every per-bin `NUFFT` and `Multiply` on each call, and
   `data_fidelity_gradient_r` builds them twice more; identical to the pattern
   fixed in `sodium_agr/operators.py`. Left as is (superseded by `sodium_agr`),
   noted here.
3. **usability** `recon_empirical_grid_and_ifft`: `# TODO: understand why factor 1.6 is needed`
   on the Kaiser-Bessel roll-off; the 1.6 compensates the KB width/oversampling
   convention of sigpy's gridding versus Jackson's formula. Left, commented.
4. The experiment scripts hard-code data paths under `/data/sodium_mr/...`,
   read a hidden `.simulation_config.json`, and depend on `pymirc`; they are
   kept untouched as the record of the paper's experiments.

## scripts/ (legacy)

1. **bug** `recon_realdata.py`, lines 290 and 301: `filt` is used but its
   definition (line 241) is commented out -> `NameError` on any run. Not fixed
   (the script also needs the missing `pynamr` package); documented.
2. **usability** `utils.py` `TriliniearKSpaceRegridder.regrid`, line 497:
   `output /= np.sqrt((128 / self._matrix_size)**3)` hard-codes a 128
   reference matrix; `reconstruct_simulated_dual_echo_tpi_na_data.py` line 206
   hard-codes `nufft_norm = 11626.0` (the norm of a specific pynufft plan).
   Both only make sense for the paper's setup; documented.
3. `preprocess_ge_na_data.py`, `scripts/utils.py` and `sodium_agr/registration.py`
   carry three copies of `align_images`; the package copy is canonical.
4. `inverse_crime_mono_exp.py`, `sim_recon_phantom.py`, `recon_realdata.py`,
   `reconstruct_ge_tpi_data.py`, `reconstruct_simulated_*` need `pynamr`,
   `pymirc`, `pynufft`, `cost_functions`, `nearest_neighbors` from paths that
   are not in the repository (`sys.path.append('..')` expects
   `bowsher-cartesian-sodium-agr` modules in the parent directory). They are
   not runnable from a clean checkout; left as historical record.

## bowsher-cartesian-sodium-agr/

1. **ineff** `cost_functions.py`: `multi_echo_bowsher_cost` and
   `multi_echo_bowsher_grad` each run the full forward model, so every L-BFGS
   iteration does the forward pass twice (cost and gradient are separate
   callbacks); `apodized_fft_multi_echo` moves `recon`, `Gam` and `sens` to the
   GPU on every call. Documented; not changed (separate workflow, not used here).
2. **bug** `apodized_fft.py`, `apodized_fft`/`adjoint_apodized_fft` (the 2D
   single-echo pair, lines 5-84): `F *= sqrt(4 ndim)/sqrt(prod(shape))` in the
   forward and `f *= sqrt(prod(shape)) * sqrt(4 ndim)` in the adjoint are not
   adjoint to each other (one divides, the other multiplies by `sqrt(prod)`);
   the multi-echo pair below uses `norm='ortho'` and is consistent. The 2D
   pair is unused by the scripts; documented, not changed.
3. **usability** `priors.py` depends on `pymirc.image_operations.grad/div`,
   not in the environment file.
4. `readout_time.py` embeds the GE TPI timing polynomial with magic constants; fine for its purpose.

## What was deliberately not changed

- The dTV formulation, the Inati coil-map algorithm, the PDHG/L-BFGS
  alternation and all default weights: the fork reproduces upstream numerics
  under `--legacy_step_sizes --legacy_init`.
- Everything under `sigpy/`, `scripts/`, `bowsher-cartesian-sodium-agr/` and
  `wip/` except the two one-line bug fixes listed above (gradient factor,
  `convert_ome_data` output).

## Questions for the owner (flags)

1. Second-echo AGR initialised with the first-echo L2 recon (master script line 479): typo or intended?
2. Ratio lower bound 0.01 (reconstruction.py line 447): intended for sodium; it must be lowered for proton UTE.
   The fork adds `--T2star_min_ms` (owner's choice for the in-vivo run: 0.05 ms, r_min = 9.6e-20 at dTE 2.19 ms).
3. `na_origin = anat_origin` (master script line 414): a GE-layout assumption; the fork makes the origin explicit.
4. The PDHG step rule: the fork's default follows the convergence condition; if the old iterates are wanted for continuity with published results, use `--legacy_step_sizes`.
