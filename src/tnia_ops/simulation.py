"""Test images.

These need raster-geometry, which is not yet ported to NumPy 2, so they run
in the numpy_1x environment.
"""

from typing import Annotated

import numpy as np
from opspec.op import op
from opspec.types import ImageOf


@op(env="numpy_1x")
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


@op(env="numpy_1x")
def random_spheres(
    depth: Annotated[int, {"min": 8, "max": 2048}] = 64,
    height: Annotated[int, {"min": 8, "max": 8192}] = 256,
    width: Annotated[int, {"min": 8, "max": 8192}] = 256,
    count: Annotated[int, {"min": 1, "max": 10000}] = 30,
    min_radius: Annotated[int, {"min": 1, "max": 500}] = 5,
    max_radius: Annotated[int, {"min": 1, "max": 500}] = 15,
    min_intensity: float = 100.0,
    max_intensity: float = 1000.0,
    seed: int = 1,
) -> ImageOf[np.ndarray]:
    """Random filled spheres in a volume, for testing 3D segmentation.

    A sphere near the edge is cut off by it.

    Args:
        depth: Number of planes.
        height: Plane height.
        width: Plane width.
        count: How many spheres.
        min_radius: Smallest radius.
        max_radius: Largest radius.
        min_intensity: Dimmest sphere.
        max_intensity: Brightest sphere.
        seed: Same seed, same volume.
    """
    import random

    from tnia.simulation.phantoms import random_sphere_phantom

    # random_sphere_phantom draws from the random module and takes no seed.
    random.seed(seed)
    return random_sphere_phantom(
        width,
        height,
        depth,
        count,
        min_radius,
        max_radius,
        min_intensity=int(min_intensity),
        max_intensity=int(max_intensity),
    )


@op(env="numpy_1x")
def grid_of_circles(
    height: Annotated[int, {"min": 8, "max": 8192}] = 256,
    width: Annotated[int, {"min": 8, "max": 8192}] = 256,
    radius: Annotated[int, {"min": 1, "max": 500}] = 10,
    spacing: Annotated[int, {"min": 2, "max": 2000}] = 40,
    border: Annotated[int, {"min": 0, "max": 2000}] = 20,
    intensity: float = 100.0,
) -> ImageOf[np.ndarray]:
    """Circles on a regular grid.

    Args:
        height: Image height.
        width: Image width.
        radius: Circle radius.
        spacing: From one centre to the next.
        border: First centre's distance from the edge; at least the radius.
        intensity: The circles' value.
    """
    from tnia.simulation.phantoms import grid_of_circles as tnia_grid_of_circles

    image = np.zeros((height, width), np.float32)
    tnia_grid_of_circles(image, radius, spacing, border, intensity)
    return image
