"""Coil sensitivity estimation (Inati et al., ISMRM 2014) and SENSE scaling.

The Inati routine is a port of ismrmrd-python-tools / Gadgetron's
``coil_map_3d_Inati_Iter``; it runs on numpy (the box smoothing dominates and
is cheap at the grid sizes used here).
"""
from __future__ import annotations

import numpy as np
import sigpy
import sigpy.mri
from scipy import ndimage

from .backend import device_of, to_device


def calculate_sense_scale(coil_maps: np.ndarray, k: np.ndarray, data: np.ndarray,
                          field_of_view_cm: float, max_iter: int = 30) -> float:
    """Scale by which the coil maps must be divided for the SENSE operator to have norm 1."""
    k_d = to_device(np.asarray(k, dtype=np.float32).reshape(-1, 3) * float(field_of_view_cm))
    S = sigpy.mri.linop.Sense(to_device(coil_maps), coord=k_d)
    max_eig = sigpy.app.MaxEig(S.H * S, dtype=np.asarray(data).dtype, device=device_of(k_d),
                               max_iter=max_iter, show_pbar=False).run()
    return float(np.sqrt(float(max_eig)))


def smooth(img: np.ndarray, box=5) -> np.ndarray:
    """Box-smooth a complex image (real and imaginary parts separately)."""
    return (ndimage.uniform_filter(img.real, size=box) + 1j * ndimage.uniform_filter(img.imag, size=box))


def calculate_csm_inati_iter(im, smoothing=5, niter: int = 5, thresh: float = 1e-3, verbose: bool = False):
    """Fast iterative coil-map estimation and adaptive coil combination.

    Parameters
    ----------
    im : (ncoils, ny, nx) or (ncoils, nz, ny, nx) complex images
    smoothing : int or 3 ints, box size of the spatial smoothing
    niter, thresh : iterations and relative-change stopping threshold

    Returns
    -------
    coil_map : relative sensitivities, same shape as ``im``
    coil_combined : the combined image
    """
    im = np.asarray(im)
    if im.ndim < 3 or im.ndim > 4:
        raise ValueError("expected 3D [ncoils, ny, nx] or 4D [ncoils, nz, ny, nx] input")
    images_are_2d = im.ndim == 3
    if images_are_2d:
        im = im[:, np.newaxis]
    smoothing = np.asarray([smoothing] * 3 if np.ndim(smoothing) == 0 else smoothing)
    if smoothing.size != 3:
        raise ValueError("smoothing should be an int or 3 ints")
    if images_are_2d:
        smoothing[2] = 1
    box = np.concatenate(([1], smoothing))
    ncha = im.shape[0]
    D_sum = im.sum(axis=(1, 2, 3))
    D_sum *= 1.0 / np.linalg.norm(D_sum)
    R = np.zeros(im.shape[1:], dtype=im.dtype)
    for cha in range(ncha):
        R += np.conj(D_sum[cha]) * im[cha]
    eps = np.finfo(im.real.dtype).eps * np.abs(im).mean()
    coil_map = None
    for it in range(niter):
        if verbose:
            print(f"coil map estimation: iteration {it + 1} of {niter}")
        prevR = R.copy() if thresh > 0 else None
        R = np.conj(R)
        coil_map = im * R[np.newaxis]
        coil_map_conv = smooth(coil_map, box=box)
        D = coil_map_conv * np.conj(coil_map_conv)
        R = 1.0 / (np.sqrt(D.sum(axis=0)) + eps)
        coil_map = coil_map_conv * R[np.newaxis]
        D = im * np.conj(coil_map)
        R = D.sum(axis=0)
        D = coil_map * R[np.newaxis]
        D_sum = D.sum(axis=(1, 2, 3))           # upstream's fallback branch summed `im` here by mistake
        D_sum *= 1.0 / np.linalg.norm(D_sum)
        imT = np.zeros(R.shape, dtype=im.dtype)
        for cha in range(ncha):
            imT += np.conj(D_sum[cha]) * coil_map[cha]
        imT /= np.abs(imT) + eps
        R = R * imT
        coil_map = coil_map * np.conj(imT)[np.newaxis]
        if thresh > 0:
            vratio = np.linalg.norm(R - prevR) / np.linalg.norm(R)
            if verbose:
                print(f"  relative change {vratio:.3e}")
            if vratio < thresh:
                break
    coil_combined = (im * np.conj(coil_map)).sum(0)
    if images_are_2d:
        coil_combined = coil_combined[0]
        coil_map = coil_map[:, 0]
    return coil_map, coil_combined
