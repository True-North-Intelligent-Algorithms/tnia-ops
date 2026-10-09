"""Thresholding and filtering objects."""

from typing import Annotated

import numpy as np
from opspec.op import op
from opspec.types import ImageOf, LabelsOf


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
