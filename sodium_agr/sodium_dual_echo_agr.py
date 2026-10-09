"""Dual-echo AGR with decay estimation and modelling: command-line entry point.

Input layout of a subject directory::

    subject_directory
    |-- anatomical_prior_image.nii
    |-- kspace_trajectory.h5            dataset k (num_points, num_readouts, 3) in 1/cm + parameters as attributes
    |-- raw_echo1/converted_data.h5     dataset data (num_coils, num_points, num_readouts), complex
    `-- raw_echo2/converted_data.h5

To build these files from GE / Siemens TPI data use the readers in
``sodium_agr.preprocessing``; for any other sequence use
``write_kspace_trajectory_h5`` / ``write_converted_data_h5`` from the same module.

All reconstructions are written to ``<subject>/recon_<N>_<N>_<N>/`` as NIfTI
(magnitude) and npz (complex image + PDHG dual variable). Re-running resumes
from the cached stages. The reconstructed image ``x`` of the final stage is
the magnetisation at t = 0 (the decay from excitation to TE1 is modelled);
``r`` is ``exp(-(TE2 - TE1) / T2*)``.

Runs on CUDA (cupy) when available, otherwise on numpy (slow; set
``SODIUM_AGR_BACKEND=numpy`` to force it).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a script from inside the folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sodium_agr.pipeline import ReconConfig, run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    d = ReconConfig(sdir="")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sdir", required=True, help="subject directory containing the raw data")
    p.add_argument("--matrix_size", type=int, default=d.matrix_size, help="recon matrix size")
    p.add_argument("--fov_cm", type=float, default=d.fov_cm, help="recon field of view (cm)")
    p.add_argument("--TE1_ms", type=float, default=d.TE1_ms, help="first echo time (ms)")
    p.add_argument("--TE2_ms", type=float, default=d.TE2_ms, help="second echo time (ms)")
    p.add_argument("--beta_non_anatomical", type=float, default=d.beta_non_anatomical, help="quadratic prior weight (SENSE init)")
    p.add_argument("--beta_anatomical", type=float, default=d.beta_anatomical, help="anatomical L1 (dTV) prior weight")
    p.add_argument("--beta_r", type=float, default=d.beta_r, help="anatomical L2 prior weight for the ratio image")
    p.add_argument("--eta", type=float, default=d.eta, help="dTV joint-gradient softening")
    p.add_argument("--max_iter_agr", type=int, default=d.max_iter_agr, help="PDHG iterations for AGR without decay modelling")
    p.add_argument("--max_outer_iter_agr", type=int, default=d.max_outer_iter_agr, help="outer iterations for AGR with decay modelling")
    p.add_argument("--inner_iter_decay", type=int, default=d.inner_iter_decay, help="PDHG iterations per outer iteration")
    p.add_argument("--lbfgs_iter", type=int, default=d.lbfgs_iter, help="L-BFGS iterations per ratio update")
    p.add_argument("--num_time_bins", type=int, default=d.num_time_bins, help="readout time bins of the decay model")
    p.add_argument("--r_bounds", type=float, nargs=2, default=list(d.r_bounds), metavar=("RMIN", "RMAX"),
                   help="bounds of the ratio image exp(-dTE/T2*); lower it for short T2* (proton UTE)")
    p.add_argument("--csm_smoothing", type=int, default=d.csm_smoothing, help="box size of the coil-map smoothing")
    p.add_argument("--preview_matrix", type=int, default=d.preview_matrix, help="matrix of the gridding preview")
    p.add_argument("--anat_file", default=d.anat_file, help="anatomical prior NIfTI inside the subject directory")
    p.add_argument("--sodium_origin_mm", type=float, nargs=3, default=None, help="world origin (mm) of the recon grid; default FOV centred at 0")
    p.add_argument("--origin_from_anat", action="store_true", help="upstream behaviour: take the grid origin from the anatomical NIfTI")
    p.add_argument("--legacy_step_sizes", action="store_true", help="reproduce upstream PDHG step sizes")
    p.add_argument("--legacy_init", action="store_true", help="reproduce upstream: initialise the echo-2 AGR with the echo-1 SENSE recon")
    p.add_argument("--save_intermediates", action="store_true", help="save r/x after every outer iteration of the decay stage")
    p.add_argument("--output_subdir", default=None, help="name of the output directory inside the subject directory")
    p.add_argument("--show_pbar", action="store_true")
    return p


def main(argv=None) -> ReconConfig:
    args = vars(build_parser().parse_args(argv))
    args["r_bounds"] = tuple(args["r_bounds"])
    if args["sodium_origin_mm"] is not None:
        args["sodium_origin_mm"] = tuple(args["sodium_origin_mm"])
    cfg = ReconConfig(**args)
    run(cfg)
    return cfg


if __name__ == "__main__":
    main()
