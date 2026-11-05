"""A bounding box around each object."""

import numpy as np
from opspec.op import op
from opspec.types import BoxesOf, LabelsOf


@op(env="skimage")
def object_boxes(labels: LabelsOf[np.ndarray]) -> BoxesOf[np.ndarray]:
    """A bounding box around each labelled object.

    Args:
        labels: The labels.
    """
    from ophelper.boxes import from_labels

    return from_labels(labels)
