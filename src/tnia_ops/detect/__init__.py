"""Finding things in images.

The same namespace as scikit-ops' ``detect``, which has the general-purpose
detectors (any YOLO checkpoint, FastSAM); these are for one kind of thing.
"""

from .bibs import Bibs, detect_bibs
from .people import KEYPOINTS, People, Poses, detect_people, detect_poses

__all__ = ["KEYPOINTS", "Bibs", "People", "Poses", "detect_bibs", "detect_people", "detect_poses"]
