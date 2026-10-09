"""Staged dual-echo AGR pipeline (the former 550-line ``sodium_dual_echo_agr.py``).

Stages, each cached in the output directory so a run can be resumed:

1. load the two echoes and the trajectory, normalise the data
2. density-compensated gridding per coil (preview), coil-combined
3. early-stopped channel-wise least squares (input for the coil maps)
4. coil sensitivities (Inati) and the SENSE scale (``||S|| = 1``)
5. SENSE with a quadratic prior, per echo (initialisation and registration target)
6. rigid alignment of the anatomical prior -> projected-gradient (dTV) operator
7. AGR (dTV, L1) per echo without decay modelling
8. joint dual-echo AGR with decay (ratio) estimation

Every number that was hard-coded upstream is a field of :class:`ReconConfig`.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk

from .backend import BACKEND_NAME, to_device, to_numpy, xp
from .coils_sens import calculate_csm_inati_iter, calculate_sense_scale
from .operators import projected_gradient_operator
from .preprocessing import read_converted_data_h5, read_kspace_trajectory_h5
from .reconstruction import (channelwise_ifft_recon, channelwise_lsq_recon,
                             dual_echo_sense_with_decay_estimation, regularized_sense_recon)
from .registration import align_images


@dataclasses.dataclass
class ReconConfig:
    """All parameters of a dual-echo AGR run (defaults reproduce upstream's values)."""
    sdir: str                                  # subject directory (see module docstring of sodium_dual_echo_agr.py)
    matrix_size: int = 128
    fov_cm: float = 22.0
    TE1_ms: float = 0.5
    TE2_ms: float = 5.0
    beta_non_anatomical: float = 1e-1          # quadratic prior weight (stage 5)
    beta_anatomical: float = 3e-3              # dTV L1 weight (stages 7, 8)
    beta_r: float = 3e-2                       # quadratic dTV weight on the ratio image (stage 8)
    eta: float = 5e-3                          # dTV joint-gradient softening
    max_iter_agr: int = 500                    # PDHG iterations, stage 7
    max_outer_iter_agr: int = 20               # outer iterations, stage 8
    inner_iter_decay: int = 100                # PDHG iterations per outer iteration, stage 8
    lbfgs_iter: int = 10                       # L-BFGS iterations per outer iteration, stage 8
    num_time_bins: int = 64
    r_bounds: tuple[float, float] = (0.01, 1.0)
    t2star_min_ms: float | None = None         # if set, the lower ratio bound becomes exp(-(TE2 - TE1) / t2star_min_ms)
    csm_smoothing: int = 7
    lsq_iter: int = 10
    preview_matrix: int = 64                   # gridding preview grid (upstream: fixed 64)
    normalize_sample: int = 1                  # k-space sample index used for the data normalisation (upstream: 1)
    sodium_origin_mm: tuple[float, float, float] | None = None   # world origin of the recon grid; None = FOV centred at 0
    origin_from_anat: bool = False             # upstream behaviour: take the origin from the anatomical NIfTI
    anat_file: str = "anatomical_prior_image.nii"
    skip_registration: bool = False            # prior already on the recon grid: resample with the identity transform only
    legacy_step_sizes: bool = False            # reproduce upstream PDHG steps
    legacy_init: bool = False                  # reproduce upstream: echo-2 AGR initialised with echo-1 SENSE recon
    save_intermediates: bool = False
    output_subdir: str | None = None           # default: recon_<matrix>_<matrix>_<matrix>
    show_pbar: bool = False

    @property
    def effective_r_bounds(self) -> tuple[float, float]:
        """Bounds of ``r = exp(-(TE2 - TE1) / T2*)``; ``t2star_min_ms`` overrides the lower one."""
        if self.t2star_min_ms is None:
            return tuple(self.r_bounds)
        r_min = float(np.exp(-(self.TE2_ms - self.TE1_ms) / self.t2star_min_ms))
        return (max(r_min, 1e-30), self.r_bounds[1])

    @property
    def t2star_bounds_ms(self) -> tuple[float, float]:
        """The T2* range the ratio bounds correspond to (upper bound 1 means infinite T2*)."""
        lo, hi = self.effective_r_bounds
        d = self.TE2_ms - self.TE1_ms
        return (-d / np.log(lo), np.inf if hi >= 1 else -d / np.log(hi))

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return (self.matrix_size,) * 3

    @property
    def subject_path(self) -> Path:
        return Path(self.sdir)

    @property
    def output_path(self) -> Path:
        sub = self.output_subdir or f"recon_{'_'.join(str(x) for x in self.grid_shape)}"
        return self.subject_path / sub

    @property
    def output_affine(self) -> np.ndarray:
        return np.diag(np.concatenate([10 * self.fov_cm / np.array(self.grid_shape), [1]]))

    @property
    def voxel_size_mm(self) -> np.ndarray:
        return 10 * self.fov_cm / np.array(self.grid_shape, dtype=float)


class Stage:
    """Tiny cache helper: ``Stage(path).get(compute)`` loads or computes-and-saves an npz."""

    def __init__(self, path: Path, verbose: bool = True) -> None:
        self.path = Path(path)
        self.verbose = verbose

    def get(self, compute):
        if self.path.exists():
            if self.verbose:
                print(f"loading {self.path}")
            with np.load(self.path, allow_pickle=False) as f:
                return {k: f[k] for k in f.files}
        result = compute()
        if self.verbose:
            print(f"saving {self.path}")
        np.savez_compressed(self.path, **result)
        return result


# ---------------------------------------------------------------------------
def load_inputs(cfg: ReconConfig):
    """Stage 1: echoes, trajectory, density compensation; data normalised to ~1."""
    data_1, rotation = read_converted_data_h5(cfg.subject_path / "raw_echo1" / "converted_data.h5")
    data_2, _ = read_converted_data_h5(cfg.subject_path / "raw_echo2" / "converted_data.h5")
    k, params = read_kspace_trajectory_h5(cfg.subject_path / "kspace_trajectory.h5")
    if rotation == 0:  # GE rotation tag: swap x/y to land in RAS
        k = np.stack([k[..., 1], -k[..., 0], k[..., 2]], axis=-1)
    k = k[: data_1.shape[1]]
    norm = np.abs(data_1[:, min(cfg.normalize_sample, data_1.shape[1] - 1), :]).max()
    data_1 = data_1 / norm
    data_2 = data_2 / norm
    abs_k = np.linalg.norm(k, axis=-1)
    dk = abs_k[1:, 0] - abs_k[:-1, 0]
    i_twist = int(np.where(dk >= 0.99 * dk[5:].max())[0].max())    # end of the linear (radial) part
    dcf = np.clip(abs_k ** 2, None, abs_k[i_twist] ** 2)
    dcf /= dcf.max()
    return data_1, data_2, k, params, dcf, float(norm)


def run(cfg: ReconConfig) -> dict:
    """Run the whole pipeline; returns the final image, ratio image and paths of the stage files."""
    out = cfg.output_path
    out.mkdir(exist_ok=True, parents=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    print(f"backend: {BACKEND_NAME}; output: {out}")
    lo, hi = cfg.effective_r_bounds; t_lo, t_hi = cfg.t2star_bounds_ms
    print(f"ratio bounds {lo:.3g} .. {hi:.3g}  (T2* {t_lo:.3f} ms .. {t_hi:.3g} ms for dTE = {cfg.TE2_ms - cfg.TE1_ms:.3f} ms)")
    aff = cfg.output_affine

    data_1, data_2, k, params, dcf, norm = load_inputs(cfg)
    grid = cfg.grid_shape
    preview = (min(cfg.preview_matrix, cfg.matrix_size),) * 3

    # stage 2: gridding preview
    st = Stage(out / "coil_combined_ifft.npz").get(lambda: _stage_ifft(data_1, data_2, k, dcf, preview, cfg))
    # stage 3: channel-wise least squares
    lsq = Stage(out / "channelwise_lsq_early.npz").get(lambda: dict(
        lsq_1=channelwise_lsq_recon(data_1, k, grid, cfg.fov_cm, max_iter=cfg.lsq_iter),
        lsq_2=channelwise_lsq_recon(data_2, k, grid, cfg.fov_cm, max_iter=cfg.lsq_iter)))
    # stage 4: coil maps
    cs = Stage(out / "coil_sens_maps.npz").get(lambda: _stage_coil_maps(lsq["lsq_1"], lsq["lsq_2"], data_1, k, cfg, aff))
    maps_1, maps_2 = cs["coil_sens_maps_1"], cs["coil_sens_maps_2"]
    # stage 5: SENSE + quadratic prior
    tag = f"b{cfg.beta_non_anatomical:.2E}"
    l2_1 = Stage(out / f"sense_non_agr_{tag}_1.npz").get(lambda: _stage_sense_l2(data_1, maps_1, k, cs["coil_combined_lsq_1"], cfg))
    l2_2 = Stage(out / f"sense_non_agr_{tag}_2.npz").get(lambda: _stage_sense_l2(data_2, maps_2, k, cs["coil_combined_lsq_2"], cfg))
    sense_L2_1, sense_L2_2 = l2_1["x"], l2_2["x"]
    # stage 6: anatomical prior alignment and projected gradient
    anat_aligned = _stage_align_prior(cfg, out, np.abs(sense_L2_1))
    PG = projected_gradient_operator(anat_aligned.astype(np.float32), eta=cfg.eta)
    # phase of the SENSE recons folded into the coil maps -> real-valued images later
    maps_1 = maps_1 * np.exp(1j * np.angle(sense_L2_1))
    maps_2 = maps_2 * np.exp(1j * np.angle(sense_L2_2))
    # stage 7: AGR without decay
    tag = f"b{cfg.beta_anatomical:.2E}"
    init_2 = sense_L2_1 if cfg.legacy_init else sense_L2_2
    agr_1 = Stage(out / f"agr_no_decay_{tag}_1.npz").get(lambda: _stage_agr(data_1, maps_1, k, PG, sense_L2_1, cfg, out, f"agr_no_decay_{tag}_1"))
    agr_2 = Stage(out / f"agr_no_decay_{tag}_2.npz").get(lambda: _stage_agr(data_2, maps_2, k, PG, init_2, cfg, out, f"agr_no_decay_{tag}_2"))
    # stage 8: joint decay estimation
    r0 = initial_ratio(agr_2["x"], agr_1["x"])
    num_data = int(np.size(data_1))
    u0 = np.concatenate((agr_1["u"][:num_data], agr_2["u"][:num_data], agr_1["u"][num_data:]))
    name = f"dual_echo_agr_w_decay_{tag}_br{cfg.beta_r:.2E}_n{cfg.max_outer_iter_agr}"
    final = Stage(out / f"{name}.npz").get(lambda: _stage_decay(data_1, data_2, params, maps_1, maps_2, k, agr_1["x"], u0, r0, PG, cfg, out, name))
    return dict(x=final["x"], r=final["r"], anat_aligned=anat_aligned, output_path=out, data_norm=norm)


# ---------------------------------------------------------------------------
# stage implementations
# ---------------------------------------------------------------------------
def _stage_ifft(data_1, data_2, k, dcf, preview, cfg):
    iffts_1 = channelwise_ifft_recon(data_1, k, dcf, preview, cfg.fov_cm)
    iffts_2 = channelwise_ifft_recon(data_2, k, dcf, preview, cfg.fov_cm)
    _, comb_1 = calculate_csm_inati_iter(iffts_1, smoothing=cfg.csm_smoothing)
    _, comb_2 = calculate_csm_inati_iter(iffts_2, smoothing=cfg.csm_smoothing)
    ifft_aff = np.diag(np.concatenate([10 * cfg.fov_cm / np.array(preview), [1]]))
    return dict(coil_combined_ifft_1=comb_1, coil_combined_ifft_2=comb_2, ifft_aff=ifft_aff)


def _stage_coil_maps(lsq_1, lsq_2, data_1, k, cfg, aff):
    if lsq_1.shape[0] > 1:
        maps_1, comb_1 = calculate_csm_inati_iter(lsq_1, smoothing=cfg.csm_smoothing)
        maps_2, comb_2 = calculate_csm_inati_iter(lsq_2, smoothing=cfg.csm_smoothing)
    else:
        maps_1 = np.ones(lsq_1.shape, dtype=lsq_1.dtype)
        maps_2 = np.ones(lsq_2.shape, dtype=lsq_2.dtype)
        comb_1, comb_2 = lsq_1[0], lsq_2[0]
    scale = calculate_sense_scale(maps_1, k, data_1, cfg.fov_cm)
    return dict(coil_sens_maps_1=maps_1 / scale, coil_sens_maps_2=maps_2 / scale,
                coil_combined_lsq_1=comb_1 * scale, coil_combined_lsq_2=comb_2 * scale,
                coil_sens_scale=np.float64(scale), output_aff=aff)


def _stage_sense_l2(data, maps, k, x0, cfg):
    x, u = regularized_sense_recon(data, maps, k, beta=cfg.beta_non_anatomical, regularization="L2",
                                   field_of_view_cm=cfg.fov_cm, x=to_device(x0.astype(data.dtype)),
                                   legacy_step_sizes=cfg.legacy_step_sizes, show_pbar=cfg.show_pbar)
    return dict(x=x, u=u)


def _stage_align_prior(cfg, out, na_img):
    aligned_path = out / f"{Path(cfg.anat_file).stem}_aligned.nii"
    if aligned_path.exists():
        print(f"loading {aligned_path}")
        return nib.as_closest_canonical(nib.load(aligned_path)).get_fdata()
    anat_nii = nib.as_closest_canonical(nib.load(cfg.subject_path / cfg.anat_file))
    anat_img = anat_nii.get_fdata()
    anat_img = anat_img / np.percentile(anat_img, 99.9)
    anat_voxsize = anat_nii.header["pixdim"][1:4]
    anat_origin = anat_nii.affine[:-1, -1]
    if cfg.origin_from_anat:
        na_origin = anat_origin.copy()
    elif cfg.sodium_origin_mm is not None:
        na_origin = np.asarray(cfg.sodium_origin_mm, dtype=float)
    else:
        na_origin = -0.5 * (np.array(cfg.grid_shape) - 1) * cfg.voxel_size_mm   # FOV centred at the world origin
    final_transform = sitk.Euler3DTransform() if cfg.skip_registration else None
    aligned, _ = align_images(na_img, anat_img, cfg.voxel_size_mm, na_origin, anat_voxsize, anat_origin,
                              final_transform=final_transform)
    print(f"saving {aligned_path}")
    nib.save(nib.Nifti1Image(aligned.astype(np.float32), cfg.output_affine), aligned_path)
    return aligned


def _stage_agr(data, maps, k, PG, x0, cfg, out, name):
    x, u = regularized_sense_recon(data, maps, k, beta=cfg.beta_anatomical, regularization="L1", G=PG,
                                   field_of_view_cm=cfg.fov_cm, x=to_device(x0), max_iter=cfg.max_iter_agr,
                                   legacy_step_sizes=cfg.legacy_step_sizes, show_pbar=cfg.show_pbar)
    nib.save(nib.Nifti1Image(np.abs(x).astype(np.float32), cfg.output_affine), out / f"{name}.nii")
    return dict(x=x, u=u)


def initial_ratio(agr_2, agr_1, low_signal_fraction: float = 0.05) -> np.ndarray:
    """``|agr_2| / |agr_1|`` clipped to [0, 1]; 1 outside the largest connected object and where ``|agr_1| = 0``."""
    a1, a2 = np.abs(agr_1), np.abs(agr_2)
    r0 = np.ones(a1.shape, dtype=np.float64)
    np.divide(a2, a1, out=r0, where=a1 > 0)
    r0 = np.clip(r0, 0, 1)
    mask = a1 >= low_signal_fraction * a1.max()
    label, _ = ndimage.label(mask)
    sizes = np.bincount(label.ravel())
    if sizes.size > 1:
        r0[label != (sizes[1:].argmax() + 1)] = 1
    return r0


def _stage_decay(data_1, data_2, params, maps_1, maps_2, k, x0, u0, r0, PG, cfg, out, name):
    x, r = dual_echo_sense_with_decay_estimation(
        data_1, data_2, params.sampling_time_us, cfg.TE1_ms, cfg.TE2_ms, maps_1, maps_2, k, x0=x0, u0=u0, r0=r0,
        G=PG, field_of_view_cm=cfg.fov_cm, regularization="L1", beta=cfg.beta_anatomical,
        max_iter=cfg.inner_iter_decay, max_outer_iter=cfg.max_outer_iter_agr, num_time_bins=cfg.num_time_bins,
        beta_r=cfg.beta_r, r_bounds=cfg.effective_r_bounds, lbfgs_iter=cfg.lbfgs_iter,
        legacy_step_sizes=cfg.legacy_step_sizes,
        intermediate_dir=(out / f"{name}_intermediates") if cfg.save_intermediates else None)
    nib.save(nib.Nifti1Image(np.abs(x).astype(np.float32), cfg.output_affine), out / f"{name}.nii")
    nib.save(nib.Nifti1Image(np.abs(r).astype(np.float32), cfg.output_affine), out / f"{name}_r.nii")
    return dict(x=x, r=r)
