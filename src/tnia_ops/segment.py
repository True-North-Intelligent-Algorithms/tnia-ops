"""Thresholding and filtering objects."""

from typing import Annotated

import numpy as np
from opspec.op import op
from opspec.runtime import progress, run
from opspec.types import BoxesOf, ImageOf, LabelsOf


@op(env="skimage")
def local_global_threshold(
    image: ImageOf[np.ndarray],
    window: Annotated[int, {"min": 3, "max": 1001}] = 51,
    global_threshold: float = 0.0,
) -> LabelsOf[np.ndarray]:
    """Pixels above both a local and a global threshold.

    Args:
        image: The image.
        window: Local window size; odd.
        global_threshold: Everything kept is above this.
    """
    from skimage.measure import label
    from tnia.segmentation.thresholding import local_global_threshold as tnia_threshold

    return label(tnia_threshold(image, window, global_threshold))


@op(env="skimage")
def find_large_objects(
    image: ImageOf[np.ndarray],
    labels: LabelsOf[np.ndarray],
    min_area: Annotated[int, {"min": 1, "max": 1000000}] = 100,
) -> LabelsOf[np.ndarray]:
    """Keep objects at least *min_area* pixels big.

    Args:
        image: The intensity image.
        labels: Objects found in it.
        min_area: Smallest kept, in pixels.
    """
    from tnia.segmentation.filterobjects import find_large_objects as tnia_find_large

    return tnia_find_large(image, labels, min_area)


@op(env="skimage")
def object_boxes(labels: LabelsOf[np.ndarray]) -> BoxesOf[np.ndarray]:
    """A bounding box around each labelled object.

    Args:
        labels: The labels.
    """
    from ophelper.boxes import from_labels

    return from_labels(labels)


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
