"""Threshold, then fill holes: a workflow."""

from typing import Annotated

import numpy as np
from opspec.op import op
from opspec.runtime import progress, run
from opspec.types import ImageOf, LabelsOf

from .local_global_threshold import local_global_threshold


@op  # no environment: runs where it's called; each step runs in its own
def threshold_and_fill(
    image: ImageOf[np.ndarray],
    window: Annotated[int, {"min": 3, "max": 1001}] = 51,
    global_threshold: float = 0.0,
    max_hole_area: Annotated[int, {"min": 1, "max": 100000}] = 500,
) -> LabelsOf[np.ndarray]:
    """Threshold, then fill holes: two ops in a row.

    Args:
        image: The image.
        window: Local window size; odd.
        global_threshold: Everything kept is above this.
        max_hole_area: Bigger holes stay open.
    """
    from tnia_ops.morphology import fill_holes

    progress("Thresholding")
    labels = run(
        local_global_threshold,
        image=image,
        window=window,
        global_threshold=global_threshold,
    )
    progress("Filling holes")
    return run(fill_holes, labels=labels, max_hole_area=max_hole_area)
