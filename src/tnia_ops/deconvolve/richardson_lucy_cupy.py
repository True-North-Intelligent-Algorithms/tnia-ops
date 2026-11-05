"""Richardson-Lucy deconvolution, on an NVIDIA GPU."""

from typing import Annotated

import numpy as np
from opspec.op import Axes, op
from opspec.types import ImageOf


@op(env="cupy")
def richardson_lucy_cupy(
    image: Annotated[ImageOf[np.ndarray], Axes("z?", "y", "x")],
    psf: np.ndarray,
    num_iters: Annotated[int, {"min": 1, "max": 1000}] = 10,
    noncirc: bool = False,
) -> ImageOf[np.ndarray]:
    """Richardson-Lucy deconvolution, on an NVIDIA GPU with cupy.

    Args:
        image: The image to deconvolve.
        psf: The point spread function.
        num_iters: How many iterations.
        noncirc: Non-circulant edge handling.
    """
    import cupy as cp
    from tnia.deconvolution.richardson_lucy import richardson_lucy_cp

    return cp.asnumpy(richardson_lucy_cp(image, psf, num_iters, noncirc=noncirc))
