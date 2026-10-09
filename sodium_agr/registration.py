"""Rigid registration of the anatomical prior to the sodium/UTE image (SimpleITK, mutual information)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import SimpleITK as sitk


def numpy_volume_to_sitk_image(vol: np.ndarray, voxel_size: Sequence[float], origin: Sequence[float]) -> sitk.Image:
    """numpy (x, y, z) volume -> sitk image with the given spacing and origin (mm)."""
    image = sitk.GetImageFromArray(np.ascontiguousarray(np.swapaxes(vol, 0, 2)))
    image.SetSpacing(np.asarray(voxel_size, dtype=np.float64).tolist())
    image.SetOrigin(np.asarray(origin, dtype=np.float64).tolist())
    return image


def sitk_image_to_numpy_volume(image: sitk.Image) -> np.ndarray:
    return np.swapaxes(sitk.GetArrayFromImage(image), 0, 2)


def align_images(fixed_image: np.ndarray, moving_image: np.ndarray,
                 fixed_voxsize: Sequence[float] = (1.0, 1.0, 1.0), fixed_origin: Sequence[float] = (0.0, 0.0, 0.0),
                 moving_voxsize: Sequence[float] = (1.0, 1.0, 1.0), moving_origin: Sequence[float] = (0.0, 0.0, 0.0),
                 final_transform: sitk.Transform | None = None, verbose: bool = True,
                 learning_rate: float = 0.2, iterations: int = 400, sampling_percentage: float = 0.1):
    """Rigidly register ``moving_image`` onto ``fixed_image`` and resample it onto the fixed grid.

    Returns ``(moving_image_aligned, transform)``. Passing ``final_transform``
    skips the optimisation and only resamples.
    """
    fixed = numpy_volume_to_sitk_image(fixed_image.astype(np.float32), fixed_voxsize, fixed_origin)
    moving = numpy_volume_to_sitk_image(moving_image.astype(np.float32), moving_voxsize, moving_origin)
    if final_transform is None:
        initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        # at least ~5000 metric samples; a fixed 10 % fails on small (test) volumes
        n_vox = float(np.prod(fixed.GetSize()))
        reg.SetMetricSamplingPercentage(min(1.0, max(sampling_percentage, 5000.0 / n_vox)))
        reg.SetInterpolator(sitk.sitkLinear)
        # upstream pyramid [4, 2, 1]; levels that would shrink the fixed image below
        # 8 voxels are dropped (the MI metric fails when every sample maps outside)
        levels = [(f, s) for f, s in ((4, 2), (2, 1), (1, 0)) if min(fixed.GetSize()) // f >= 8]
        reg.SetShrinkFactorsPerLevel(shrinkFactors=[f for f, _ in levels])
        reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[s for _, s in levels])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        # upstream's learning rate (0.2) can throw the moving image completely out of overlap on a coarse
        # fixed image ("All samples map outside moving image buffer"); retry with smaller steps
        for attempt, lr in enumerate([learning_rate / 2 ** i for i in range(4)]):
            reg.SetOptimizerAsGradientDescentLineSearch(learningRate=lr, numberOfIterations=iterations,
                                                        convergenceMinimumValue=1e-7, convergenceWindowSize=10)
            reg.SetOptimizerScalesFromPhysicalShift()
            reg.SetInitialTransform(initial, inPlace=False)
            try:
                final_transform = reg.Execute(sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32))
                break
            except RuntimeError as err:
                if "map outside" not in str(err) or attempt == 3:
                    raise
                if verbose:
                    print(f"registration diverged with learning rate {lr:.3g}; retrying with {lr / 2:.3g}")
        if verbose:
            print(f"registration stopped: {reg.GetOptimizerStopConditionDescription()}")
            print(f"final metric value: {reg.GetMetricValue()}")
            print(f"final parameters: {final_transform.GetParameters()}")
    resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    return sitk_image_to_numpy_volume(resampled), final_transform
