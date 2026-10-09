"""Point spread functions."""

from typing import Annotated

import numpy as np
from opspec.op import op
from opspec.types import ImageOf


@op(env="skimage")
def gaussian_psf(
    xy_size: Annotated[int, {"min": 3, "max": 512}] = 64,
    z_size: Annotated[int, {"min": 1, "max": 512}] = 32,
    xy_sigma: Annotated[float, {"min": 0.1, "max": 50.0}] = 2.0,
    z_sigma: Annotated[float, {"min": 0.1, "max": 50.0}] = 4.0,
) -> ImageOf[np.ndarray]:
    """A 3D Gaussian PSF.

    Args:
        xy_size: Width and height, in pixels.
        z_size: Depth, in pixels.
        xy_sigma: Width of the Gaussian in x and y.
        z_sigma: Width of the Gaussian in z.
    """
    from tnia.deconvolution.gaussian_psf import gaussian_3d

    return gaussian_3d(xy_size, z_size, xy_sigma, z_sigma)


@op(env="skimage")
def paraxial_psf(
    size: Annotated[int, {"min": 8, "max": 1024}] = 128,
    wavelength: Annotated[float, {"min": 0.2, "max": 2.0}] = 0.5,
    numerical_aperture: Annotated[float, {"min": 0.1, "max": 1.7}] = 1.4,
    pixel_size: Annotated[float, {"min": 0.001, "max": 10.0}] = 0.05,
) -> ImageOf[np.ndarray]:
    """A 2D paraxial PSF, summing to 1.

    Args:
        size: Width and height, in pixels.
        wavelength: Emission wavelength, in microns.
        numerical_aperture: The objective's NA.
        pixel_size: In microns.
    """
    from tnia.deconvolution.psfs import paraxial_psf as tnia_paraxial_psf

    return tnia_paraxial_psf(size, wavelength, numerical_aperture, pixel_size)
