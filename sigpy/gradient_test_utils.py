from __future__ import annotations

import numpy as np
import cupy as cp
import nibabel as nib

from pathlib import Path
from pymirc.image_operations import zoom3d
from scipy.ndimage import binary_closing, zoom

def cost_and_grad_wrapper(x, A, y, G, beta):
    """cost and gradient wrapper for L-BFGS-B optimization
       using the L2-squared data fidelity and the L2-squared of the gradient

    Parameters
    ----------
    x : flat pseudo complex numpy array
        the image
    A : sigpy.linop
        forwad operator
    y : cupy array
        the data
    G : sigpy.linop
        the gradient operator
    beta : float
        weight of the prior

    Returns
    -------
    cost, gradient (in pseudo complex flat numpy array)
    """

    # convert flat pseudo complex array to complex
    x = np.squeeze(x.view(dtype=np.complex128))
    # reshape flat array to image shape and send to GPU
    x = cp.asarray(x.reshape(A.ishape))
    
    #-------------------------------------------------------------------------
    # calculate the data fidelity
    delta = A(x) - y
    data_fidelity = 0.5*float((delta*delta.conj()).real.sum())
    
    gradient_img = G(x)
    prior = 0.5*float((gradient_img.real**2 + gradient_img.imag**2).sum())
    
    cost = data_fidelity + beta * prior
    
    # calculate the gradient
    grad_data_fidelity = A.H(delta)
    grad_prior = G.H(G(x))  # gradient of 0.5*||Gx||^2 is G^H G x (upstream had an extra 0.5)
    grad = grad_data_fidelity + beta * grad_prior
    
    # send complex gradient image to CPU
    grad = cp.asnumpy(grad)
    # convert complex array back to pseudo complex and flatten
    grad  = grad.view('(2,)float').ravel()

    return cost, grad

def cos_im_for_tpi(im_shape: tuple) -> np.ndarray:
    """ Build 1D cosine based on hard coded maximum TPI frequency, field of view,
        and given input image cubic matrix size
    """

    kmax_1_cm = 1.45
    field_of_view_cm = 22.
    n = im_shape[-1]

    # spatial frequency of the cosine: N - 1 (one before maximum discretized frequency)
    #real_freq = kmax_1_cm - 1. / field_of_view_cm  # 1/cm
    
    # cosine freq. where we should still see a SNR gain at higher freq.
    real_freq = 1.3

    vox_size = field_of_view_cm / n
    period = 1. / real_freq
    period_nb_vox = period / vox_size
    # cosine frequency
    freq = 1. / period_nb_vox

    # 1d cosine in space
    cos_1d = np.cos(2 * np.pi * freq * np.linspace(0, n, n, endpoint=False))
    # cosine image
    cos_im = np.broadcast_to(cos_1d, im_shape)

    return cos_im


def setup_gradient_brainweb_phantom(
    simulation_matrix_size: int,
    phantom_data_path: Path,
    field_of_view_cm: float = 22.,
    csf_na_concentration: float = 3.0,
    gm_na_concentration: float = 1.5,
    wm_na_concentration: float = 1.0,
    other_na_concentration: float = 0.75,
    T2long_ms_csf: float = 50.,
    T2long_ms_gm: float = 15.,
    T2long_ms_wm: float = 18.,
    T2long_ms_other: float = 15.,
    T2short_ms_csf: float = 50.,
    T2short_ms_gm: float = 8.,
    T2short_ms_wm: float = 9.,
    T2short_ms_other: float = 8.,
    add_anatomical_mismatch: bool = False,
    add_T2star_bias: bool = False):

    simulation_voxel_size_mm: float = 10 * field_of_view_cm / simulation_matrix_size

    # setup the phantom on a high resolution grid (0.5^3mm) first
    label_nii = nib.load(phantom_data_path / 'subject54_crisp_v.nii')
    label_nii = nib.as_closest_canonical(label_nii)

    # pad to 220mm FOV
    lab_voxelsize = label_nii.header['pixdim'][1]
    lab = label_nii.get_fdata()
    pad_size_220 = ((220 - np.array(lab.shape) * lab_voxelsize) /
                    lab_voxelsize / 2).astype(int)
    pad_size_220 = ((pad_size_220[0], pad_size_220[0]),
                    (pad_size_220[1], pad_size_220[1]), (pad_size_220[2],
                                                         pad_size_220[2]))
    lab = np.pad(lab, pad_size_220, 'constant')

    roi_image = np.zeros(lab.shape, dtype=np.uint8)

    # CSF = 1, GM = 2, WM = 3
    csf_inds = np.where(lab == 1)
    gm_inds = np.where(lab == 2)
    wm_inds = np.where(lab == 3)
    other_inds = np.where(lab >= 4)
    skull_inds = np.where(lab == 7)

    roi_image[csf_inds] = 1
    roi_image[gm_inds] = 2
    roi_image[wm_inds] = 3

    # calculate eye masks
    x = np.arange(lab.shape[0])
    X, Y, Z = np.meshgrid(x, x, x)
    R1 = np.sqrt((X - 368)**2 + (Y - 143)**2 + (Z - 97)**2)
    R2 = np.sqrt((X - 368)**2 + (Y - 291)**2 + (Z - 97)**2)
    eye1_inds = np.where((R1 < 25))
    eye2_inds = np.where((R2 < 25))

    roi_image[eye1_inds] = 4
    roi_image[eye2_inds] = 5

    # set up array for trans. magnetization
    img = np.zeros(lab.shape, dtype=np.float32)
    img[csf_inds] = csf_na_concentration
    img[gm_inds] = gm_na_concentration
    img[wm_inds] = wm_na_concentration
    img[other_inds] = other_na_concentration
    img[skull_inds] = 0.1
    img[eye1_inds] = csf_na_concentration
    img[eye2_inds] = csf_na_concentration

    # set up array for Gamma (ratio between 2nd and 1st echo)
    T2short_ms = np.full(lab.shape,
                         0.5 * np.finfo(np.float32).max,
                         dtype=np.float32)
    T2short_ms[csf_inds] = T2short_ms_csf
    T2short_ms[gm_inds] = T2short_ms_gm
    T2short_ms[wm_inds] = T2short_ms_wm
    T2short_ms[other_inds] = T2short_ms_other
    T2short_ms[eye1_inds] = T2short_ms_csf
    T2short_ms[eye2_inds] = T2short_ms_csf

    T2long_ms = np.full(lab.shape,
                        0.5 * np.finfo(np.float32).max,
                        dtype=np.float32)
    T2long_ms[csf_inds] = T2long_ms_csf
    T2long_ms[gm_inds] = T2long_ms_gm
    T2long_ms[wm_inds] = T2long_ms_wm
    T2long_ms[other_inds] = T2long_ms_other
    T2long_ms[eye1_inds] = T2long_ms_csf
    T2long_ms[eye2_inds] = T2long_ms_csf

    # read the T1
    t1_nii = nib.load(phantom_data_path / 'subject54_t1w_p4_resampled.nii')
    t1_nii = nib.as_closest_canonical(t1_nii)
    t1 = np.pad(t1_nii.get_fdata(), pad_size_220, 'constant')

    # add eye contrast
    t1[eye1_inds] *= 0.5
    t1[eye2_inds] *= 0.5

    #------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------
    # add a glioma lesion
    glioma_mask = np.fromfile(phantom_data_path / 'glioma_labshape.raw', int).reshape(img.shape)
    img[glioma_mask==1] = 0.9 * csf_na_concentration
    glioma_ring = 0.5 * wm_na_concentration
    img[glioma_mask==2] = glioma_ring
    # modify T2 values, same for before and after treatment
    T2short_ms[glioma_mask==1] = T2short_ms_csf
    T2long_ms[glioma_mask==1] = T2long_ms_csf
    T2short_ms[glioma_mask==2] = T2short_ms_wm
    T2long_ms[glioma_mask==2] = T2long_ms_wm

    roi_image[glioma_mask==1] = 6
    roi_image[glioma_mask==2] = 7
    #------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------

    # add a cosine lesion to WM
    cos_img = cos_im_for_tpi(img.shape)

    cos_mask = (X > 220) * (X<309) * (Y>220) * (Z>214) * (Z<278) * ((lab==2) + (lab==3))
    cos_mask2 = binary_closing(cos_mask, iterations=3)

    # overwrite the Na concentrations and T2 values to WM values in the cos mask
    img[cos_mask2] = wm_na_concentration
    T2short_ms[cos_mask2==1] = T2short_ms_wm
    T2long_ms[cos_mask2==1] = T2long_ms_wm

    # add the cos lesion
    img += cos_mask2 * (cos_img * wm_na_concentration + wm_na_concentration)

    roi_image[cos_mask2==1] = 8

    # add mismatches
    if add_anatomical_mismatch:
        R1 = np.sqrt((X - 329)**2 + (Y - 165)**2 + (Z - 200)**2)
        inds1 = np.where((R1 < 10))
        img[inds1] = gm_na_concentration
        #R2 = np.sqrt((X - 327)**2 + (Y - 262)**2 + (Z - 200)**2)
        #inds2 = np.where((R2 < 10))
        #t1[inds2] = 0
        roi_image[inds1] = 0

    # add bias field on T2* times
    if add_T2star_bias:
        T2starbias = np.arctan((Z - 155) / 10) / (2 * np.pi) + 0.75
        T2short_ms *= T2starbias
        T2long_ms *= T2starbias

    # extrapolate the all images to the voxel size we need for the data simulation
    img_extrapolated = zoom3d(img, lab_voxelsize / simulation_voxel_size_mm)
    T2short_ms_extrapolated = zoom3d(T2short_ms,
                                     lab_voxelsize / simulation_voxel_size_mm)
    T2long_ms_extrapolated = zoom3d(T2long_ms,
                                    lab_voxelsize / simulation_voxel_size_mm)
    t1_extrapolated = zoom3d(t1, lab_voxelsize / simulation_voxel_size_mm)

    roi_image = zoom(roi_image, lab_voxelsize / simulation_voxel_size_mm, order=0)

    return img_extrapolated, t1_extrapolated, T2short_ms_extrapolated, T2long_ms_extrapolated, roi_image

