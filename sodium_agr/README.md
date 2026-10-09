# Anatomically-guided reconstruction (AGR) of dual echo sodium TPI including T2\* decay estimation and modeling

### Installation

Install conda (or mamba). All dependencies that are needed to run the reconstructions are available on conda-forge.

Create a new virtual conda environment that contains all our dependencies

```
conda env create -f environment.yml
```

Reconstructions run on a CUDA GPU when `cupy` is installed (not part of the
environment file; install the build matching your CUDA, e.g. `pip install
cupy-cuda12x`). Without `cupy` the package falls back to numpy, which is
correct but roughly 50-100x slower; `SODIUM_AGR_BACKEND=numpy` forces the CPU
even when a GPU is present.

### Package layout

| module | content |
| --- | --- |
| `backend.py` | numpy/cupy selection (`xp`, `to_device`, `to_numpy`) |
| `operators.py` | `ApodizedNUFFT` (time-binned decay-modelled NUFFT-SENSE), gradient / projected-gradient (dTV) operators, Bowsher gradient |
| `reconstruction.py` | gridding, channel-wise LSQ, regularised SENSE (PDHG), joint dual-echo AGR with decay estimation |
| `coils_sens.py` | Inati coil maps, SENSE scale |
| `registration.py` | rigid MI registration of the anatomical prior (SimpleITK) |
| `preprocessing.py` | GE TPI readers and the hdf5 writers/readers of the input layout below |
| `pipeline.py` | `ReconConfig` + `run`: the eight cached stages |
| `sodium_dual_echo_agr.py` | command-line entry point |

Unit tests: `SODIUM_AGR_BACKEND=numpy pytest tests/test_sodium_agr.py` (from the repository root).

`--T2star_min_ms` sets the lower bound of the ratio image from the shortest T2*
you want to allow (`r_min = exp(-(TE2 - TE1) / T2star_min_ms)`); upstream's fixed
0.01 corresponds to 0.48 ms for a 2.19 ms echo spacing.

Re-running the master script resumes from the cached stage files in
`<subject>/recon_<N>_<N>_<N>/`; delete a stage file to recompute it and
everything after it. `--legacy_step_sizes --legacy_init` reproduce the
numerics of the original script (see `REVIEW_JOURNAL.md` at the repository
root for what changed and why).

---

### Running recontructions

Activate the new environment

```
conda activate sodium_mr_agr
```

Reconstruction can be run using the `sodium_dual_echo_agr.py` master script. To see all the command line options use:

```
python sodium_dual_echo_agr.py -h
```

**The top section of `sodium_dual_echo_agr.py` explains the expected layout of the input data.**

---

### Data preprocessing

To preprocess raw non-uniform kspace data into the expected hdf5 file format, use the functions defined in the `preprocessing` module.
See the top section of `sodium_dual_echo_agr.py` for more details.

**Structure of kspace_trajectory.h5**

The hdf5 file containing the coordinates of the kspace trajectory
should have the following structure.
It mainly consists of the dataset `k` containing an array of shape
`(num_time_points, num_readouts, 3)`.

```
HDF5 "kspace_trajectory.h5" {
GROUP "/" {
   DATASET "k" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( 3633, 1596, 3 ) / ( 3633, 1596, 3 ) }
      ATTRIBUTE "gamma_over_2pi_MHz_T" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SCALAR
         DATA {
         (0): 11.262
         }
      }
      ATTRIBUTE "max_grad_G_cm" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SCALAR
         DATA {
         (0): 0.16
         }
      }
      ATTRIBUTE "num_cones" {
         DATATYPE  H5T_STD_I64LE
         DATASPACE  SCALAR
         DATA {
         (0): 28
         }
      }
      ATTRIBUTE "num_points" {
         DATATYPE  H5T_STD_I64LE
         DATASPACE  SCALAR
         DATA {
         (0): 3633
         }
      }
      ATTRIBUTE "num_readouts_per_cone" {
         DATATYPE  H5T_STD_I64LE
         DATASPACE  SIMPLE { ( 28 ) / ( 28 ) }
         DATA {
         (0): 90, 88, 88, 88, 86, 86, 84, 82, 80, 78, 74, 72, 68, 64, 62, 58,
         (16): 54, 50, 46, 40, 36, 32, 28, 22, 18, 12, 8, 2
         }
      }
      ATTRIBUTE "sampling_time_us" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SCALAR
         DATA {
         (0): 10
         }
      }
   }
}
}
```

**Structure of converted_data.h5**

The structure of the converted raw data files is listed below.
It mainly contains an array of shape (num_coil_channels, num_time_points, num_readouts).

```
HDF5 "converted_data.h5" {
GROUP "/" {
   DATASET "data" {
      DATATYPE  H5T_COMPOUND {
         H5T_IEEE_F32LE "r";
         H5T_IEEE_F32LE "i";
      }
      DATASPACE  SIMPLE { ( 1, 3632, 1596 ) / ( 1, 3632, 1596 ) }
      ATTRIBUTE "hdr_files" {
         DATATYPE  H5T_STRING {
            STRSIZE H5T_VARIABLE;
            STRPAD H5T_STR_NULLTERM;
            CSET H5T_CSET_UTF8;
            CTYPE H5T_C_S1;
         }
         DATASPACE  SIMPLE { ( 1 ) / ( 1 ) }
         DATA {
         (0): "meas_64759.pwr.0.hdr"
         }
      }
   }
}
}
```
