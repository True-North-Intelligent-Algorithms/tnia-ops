"""Cleaning up and joining labels."""

from typing import Annotated

import numpy as np
from opspec.op import Axes, op
from opspec.types import LabelsOf


@op(env="skimage")
def fill_holes(
    labels: LabelsOf[np.ndarray],
    max_hole_area: Annotated[int, {"min": 1, "max": 100000}] = 500,
) -> LabelsOf[np.ndarray]:
    """Fill holes inside each labelled object.

    Args:
        labels: The labels.
        max_hole_area: Bigger holes stay open.
    """
    from tnia.morphology.fill_holes import fill_holes as tnia_fill_holes

    return tnia_fill_holes(labels, area_threshold=max_hole_area)


@op(env="skimage")
def connect_2d_in_3d(
    labels: Annotated[LabelsOf[np.ndarray], Axes("z", "y", "x")],
    max_distance: Annotated[float, {"min": 0.0, "max": 1000.0}] = 10.0,
) -> LabelsOf[np.ndarray]:
    """Join 2D labels in neighbouring planes into 3D objects.

    Args:
        labels: A stack of 2D labels.
        max_distance: Furthest join, in pixels.
    """
    from tnia.morphology.connect import connect_2d_in_3d as tnia_connect

    return tnia_connect(labels, max_distance)
