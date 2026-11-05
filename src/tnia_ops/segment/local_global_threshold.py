"""Pixels above both a local and a global threshold."""

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
