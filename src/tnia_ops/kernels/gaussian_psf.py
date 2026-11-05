"""A Gaussian point spread function."""

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
