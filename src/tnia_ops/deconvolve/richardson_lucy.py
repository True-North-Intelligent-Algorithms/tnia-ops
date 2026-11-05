"""Richardson-Lucy deconvolution, on the CPU."""

from typing import Annotated

import numpy as np
from opspec.op import Axes, op
from opspec.types import ImageOf


@op(env="skimage")
def richardson_lucy(
    image: Annotated[ImageOf[np.ndarray], Axes("z?", "y", "x")],
    psf: np.ndarray,
    num_iters: Annotated[int, {"min": 1, "max": 1000}] = 10,
    noncirc: bool = False,
) -> ImageOf[np.ndarray]:
    """Richardson-Lucy deconvolution, with numpy.

    Args:
        image: The image to deconvolve.
        psf: The point spread function.
        num_iters: How many iterations.
        noncirc: Non-circulant edge handling.
    """
    from tnia.deconvolution.richardson_lucy_np import richardson_lucy_np

    return richardson_lucy_np(image, psf, num_iters, noncirc=noncirc)
