"""Test images."""

from typing import Annotated

import numpy as np
from opspec.op import op
from opspec.types import ImageOf


@op(env="skimage")
def random_circles(
    height: Annotated[int, {"min": 8, "max": 8192}] = 512,
    width: Annotated[int, {"min": 8, "max": 8192}] = 512,
    count: Annotated[int, {"min": 1, "max": 10000}] = 50,
    min_radius: Annotated[int, {"min": 1, "max": 500}] = 5,
    max_radius: Annotated[int, {"min": 1, "max": 500}] = 20,
    min_intensity: float = 100.0,
    max_intensity: float = 1000.0,
    seed: int = 1,
) -> ImageOf[np.ndarray]:
    """Random filled circles, for testing segmentation.

    Args:
        height: Image height.
        width: Image width.
        count: How many circles.
        min_radius: Smallest radius.
        max_radius: Largest radius.
        min_intensity: Dimmest circle.
        max_intensity: Brightest circle.
        seed: Same seed, same image.
    """
    from tnia.simulation.phantoms import random_circles as tnia_random_circles

    image = np.zeros((height, width), np.float32)
    tnia_random_circles(
        image, count, min_radius, max_radius, min_intensity, max_intensity, seed
    )
    return image
