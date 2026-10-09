"""Readers and writers for the k-space trajectory and converted raw data files.

File layout expected by the pipeline (see ``sodium_dual_echo_agr.py``):

* ``kspace_trajectory.h5``: dataset ``k`` of shape ``(num_points, num_readouts, 3)``
  in 1/cm with the :class:`TPIParameters` fields as attributes.
* ``raw_echoN/converted_data.h5``: dataset ``data`` of shape
  ``(num_coils, num_points, num_readouts)``, complex or a compound ``r``/``i``
  (``real``/``imag``) dtype.

Readers exist for GE ak-wav and Siemens grdb TPI gradient files and for Siemens
ome raw data; :func:`write_kspace_trajectory_h5` and :func:`write_converted_data_h5`
let any other sequence produce the same layout.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pydantic


class TPIParameters(pydantic.BaseModel):
    sampling_time_us: float
    num_points: int
    num_cones: int
    num_readouts_per_cone: list
    max_grad_G_cm: float
    gamma_over_2pi_MHz_T: float

    @property
    def max_grad_T_cm(self) -> float:
        return self.max_grad_G_cm * 1e-4


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
def write_kspace_trajectory_h5(k: np.ndarray, params: TPIParameters | dict, output_file: str | Path,
                               compression_opts: int = 4) -> None:
    """Write a ``(num_points, num_readouts, 3)`` trajectory in 1/cm with its parameters."""
    attrs = params.model_dump() if isinstance(params, TPIParameters) else dict(params)
    with h5py.File(output_file, "w") as f:
        dset = f.create_dataset("k", data=np.asarray(k), compression="gzip", compression_opts=compression_opts)
        for key, value in attrs.items():
            dset.attrs[key] = value


def write_converted_data_h5(data: np.ndarray, output_file: str | Path, compression_opts: int = 1,
                            source_names: list[str] | None = None) -> None:
    """Write ``(num_coils, num_points, num_readouts)`` complex data as ``converted_data.h5``."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        dset = f.create_dataset("data", data=np.asarray(data), compression="gzip", compression_opts=compression_opts)
        if source_names:
            dset.attrs["hdr_files"] = list(source_names)


def read_converted_data_h5(path: str | Path) -> tuple[np.ndarray, int | None]:
    """Read ``converted_data.h5``; returns complex ``(num_coils, num_points, num_readouts)`` and the GE rotation tag (or None)."""
    with h5py.File(path, "r") as f:
        data = f["data"][:]
        rotation = int(f["header"]["rdb_hdr"]["rotation"][0]) if "header" in f else None
    if not np.iscomplexobj(data):
        names = data.dtype.names or ()
        if "real" in names:
            data = data["real"] + 1j * data["imag"]
        elif "r" in names:
            data = data["r"] + 1j * data["i"]
        else:
            raise ValueError(f"cannot interpret non-complex data dtype {data.dtype}")
    if data.ndim == 2:
        data = data[np.newaxis]
    return data, rotation


def read_kspace_trajectory_h5(path: str | Path) -> tuple[np.ndarray, TPIParameters]:
    with h5py.File(path, "r") as f:
        k = f["k"][...]
        attrs = {key: (value.tolist() if hasattr(value, "tolist") else value) for key, value in f["k"].attrs.items()}
    return k, TPIParameters(**attrs)


# ---------------------------------------------------------------------------
# GE ak-wav
# ---------------------------------------------------------------------------
def read_tpi_akwav_kspace_trajectory(fname: str, gamma_over_2pi_MHz_T: float = 11.262,
                                     output_file: str | None = None) -> tuple[np.ndarray, TPIParameters]:
    """Read GE waveforms in the external ak-wav format and integrate to k-space (1/cm)."""
    offset = 0
    desc = np.fromfile(fname, dtype=np.int8, offset=offset, count=256)
    offset += desc.size * desc.itemsize
    gpts = np.fromfile(fname, dtype=np.dtype(">u2"), offset=offset, count=1)[0]
    offset += 2
    groups = np.fromfile(fname, dtype=np.dtype(">u2"), offset=offset, count=1)[0]
    offset += 2
    intl = np.fromfile(fname, dtype=np.dtype(">u2"), offset=offset, count=groups)
    offset += intl.size * intl.itemsize
    nparams = np.fromfile(fname, dtype=np.dtype(">u2"), offset=offset, count=1)[0]
    offset += 2
    params = np.fromfile(fname, dtype=np.dtype(">f8"), offset=offset, count=nparams)
    offset += params.size * params.itemsize
    wave = np.fromfile(fname, dtype=np.dtype(">i2"), offset=offset)
    grad = np.swapaxes(wave.reshape((groups, intl[0], gpts)), 0, 2).astype(np.float32)
    grad[-1] = 0  # stop bit
    dt_us = float(params[7])
    max_grad_G_cm = float(params[3])
    grads_T_cm = grad * (max_grad_G_cm / (32767 * 100 * 100))
    k_1_cm = np.cumsum(grads_T_cm, axis=0) * dt_us * gamma_over_2pi_MHz_T
    p = TPIParameters(sampling_time_us=dt_us, num_points=int(gpts), num_cones=1,
                      num_readouts_per_cone=[int(intl[0])], max_grad_G_cm=max_grad_G_cm,
                      gamma_over_2pi_MHz_T=gamma_over_2pi_MHz_T)
    if output_file is not None:
        write_kspace_trajectory_h5(k_1_cm, p, output_file, compression_opts=9)
    return k_1_cm, p


# ---------------------------------------------------------------------------
# Siemens grdb
# ---------------------------------------------------------------------------
def read_grdb(fname: str, gamma_over_2pi_MHz_T: float = 11.262, verbose: bool = False) -> tuple[np.ndarray, TPIParameters]:
    """Read one TPI grdb gradient file (one axis) and integrate to k-space (not rotated), 1/cm."""
    params = np.fromfile(fname, dtype=np.int16, count=6)
    num_cones, num_points = int(params[0]), int(params[1])
    dt_us = float(params[2])
    max_grad_G_cm = float(params[3]) / 100
    if verbose:
        print(f"num_cones {num_cones}, num_points {num_points}, dt_us {dt_us}, max_grad_G_cm {max_grad_G_cm}")
    rpc = np.fromfile(fname, dtype=np.int16, count=num_cones, offset=params.itemsize * params.size)
    grads = np.fromfile(fname, dtype=np.int16, count=num_cones * num_points,
                        offset=params.itemsize * params.size + rpc.itemsize * rpc.size).reshape(num_cones, num_points).astype(float)
    grads[:, -1] += 1  # put the EOS bit back
    grads_T_cm = grads * ((1e-4) * max_grad_G_cm / 32768)
    k = np.cumsum(grads_T_cm, axis=1) * dt_us * gamma_over_2pi_MHz_T
    p = TPIParameters(sampling_time_us=dt_us, num_points=num_points, num_cones=num_cones,
                      num_readouts_per_cone=rpc.tolist(), max_grad_G_cm=max_grad_G_cm,
                      gamma_over_2pi_MHz_T=gamma_over_2pi_MHz_T)
    return k, p


def rotate_grdb_tpi_k(kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, num_readouts_per_cone: list[int]) -> np.ndarray:
    """Rotate the per-cone base readouts about z and add the mirrored lower half sphere.

    Returns ``(3, num_readouts_total, num_points)``.
    """
    out = []
    for i, num_readouts in enumerate(num_readouts_per_cone):
        phis = np.linspace(0, 2 * np.pi, num_readouts // 2, endpoint=False)
        k = np.zeros((3, 2 * phis.shape[0], kx.shape[1]))
        k0 = np.outer(np.cos(phis), kx[i]) - np.outer(np.sin(phis), ky[i])
        k1 = np.outer(np.sin(phis), kx[i]) + np.outer(np.cos(phis), ky[i])
        k2 = np.outer(np.ones_like(phis), kz[i])
        k[0, ::2], k[1, ::2], k[2, ::2] = -k0, k1, -k2
        k[0, 1::2], k[1, 1::2], k[2, 1::2] = k0, -k1, k2
        out.append(k)
    return np.concatenate(out, axis=1)


def read_tpi_grdb_kspace_trajectory(fname_x: str, gamma_over_2pi_MHz_T: float = 11.262,
                                    output_file: str | None = None) -> tuple[np.ndarray, TPIParameters]:
    """Read the x/y/z grdb files of a TPI trajectory and return ``(num_points, num_readouts, 3)`` in 1/cm."""
    kx, px = read_grdb(fname_x, gamma_over_2pi_MHz_T)
    ky, py = read_grdb(fname_x.replace(".x.grdb", ".y.grdb"), gamma_over_2pi_MHz_T)
    kz, pz = read_grdb(fname_x.replace(".x.grdb", ".z.grdb"), gamma_over_2pi_MHz_T)
    if px != py or px != pz:
        raise ValueError("gradient parameters differ between the x, y and z files")
    k = np.transpose(rotate_grdb_tpi_k(kx, ky, kz, px.num_readouts_per_cone), (2, 1, 0))
    if output_file is not None:
        write_kspace_trajectory_h5(k, px, output_file, compression_opts=9)
    return k, px


# ---------------------------------------------------------------------------
# Siemens ome raw data
# ---------------------------------------------------------------------------
def convert_ome_data(raw_data_dir: str, hdr_pattern: str = "*.?.hdr", output_file: str | None = None) -> np.ndarray:
    """Convert Siemens ome/hdr per-coil raw files to ``converted_data.h5``.

    Upstream only wrote the file when ``output_file`` was None (default name);
    an explicit path silently wrote nothing. Both cases write now.
    """
    hdr_files = sorted(Path(raw_data_dir).glob(hdr_pattern))
    if not hdr_files:
        raise FileNotFoundError(f"no files matching {hdr_pattern} in {raw_data_dir}")
    data = None
    for i, hdr_file in enumerate(hdr_files):
        hdr = hdr_file.read_text().splitlines()
        dim0 = int(next(x for x in hdr if "dim0" in x).split("dim0")[-1])
        dim1 = int(next(x for x in hdr if "dim1" in x).split("dim1")[-1])
        tmp = np.fromfile(hdr_file.with_suffix(".ome"), dtype=np.complex64).reshape(dim1, dim0).T
        if data is None:
            data = np.zeros((len(hdr_files),) + tmp.shape, dtype=tmp.dtype)
        data[i] = tmp
    if output_file is None:
        output_file = str(Path(raw_data_dir) / "converted_data.h5")
    write_converted_data_h5(data, output_file, compression_opts=1, source_names=[x.name for x in hdr_files])
    return data
