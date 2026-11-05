"""Finding things in images.

The same namespace as scikit-ops' ``detect``, which has the general-purpose
detectors (any YOLO checkpoint, FastSAM); these are for one kind of thing.
"""

from .people import People, detect_people

__all__ = ["People", "detect_people"]
