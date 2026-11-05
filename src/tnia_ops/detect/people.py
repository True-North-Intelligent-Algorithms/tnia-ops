"""Finding the people in a photo.

``detect_people`` is a COCO YOLO checkpoint kept to its person class, at a
chosen inference size. scikit-ops' ``yolo`` runs any checkpoint, but at the
size it was trained at (640 for COCO's), which loses people far from the
camera in a 4000-pixel photo; this runs the whole photo at 1280 instead.
"""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
from opspec.op import Axes, op
from opspec.runtime import progress
from opspec.types import BoxesOf, ImageOf

from .._torch import device

RGB = Annotated[ImageOf[np.ndarray], Axes("y", "x", "c")]

_models: dict = {}


class People(NamedTuple):
    #: (N, 4) as [min_y, min_x, max_y, max_x], tallest first.
    boxes: BoxesOf[np.ndarray]
    #: One detection confidence per box.
    confidences: list[float]


@op(env="photo")
def detect_people(
    image: RGB,
    weights: Path,
    size: Annotated[int, {"min": 320, "max": 4096}] = 1280,
    conf: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.35,
    keep_relative_height: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.45,
) -> People:
    """Find the people in a photo, the nearest ones.

    Args:
        image: RGB uint8.
        weights: A COCO-trained YOLO detection checkpoint, e.g. yolo11m.pt
            (https://github.com/ultralytics/assets/releases).
        size: Inference size, in pixels, of the photo's long side.
        conf: Minimum detection confidence.
        keep_relative_height: Drop people shorter than this fraction of the
            tallest one -- those further back. 0 keeps everyone.
    """
    from ultralytics import YOLO

    key = (str(Path(weights).resolve()), device())
    if key not in _models:
        progress(f"Loading {Path(weights).name}")
        _models[key] = YOLO(str(weights))
    progress("Finding people")
    result = _models[key].predict(
        np.ascontiguousarray(image[:, :, ::-1]),  # ultralytics takes BGR arrays
        imgsz=size,
        conf=conf,
        classes=[0],  # COCO's person
        verbose=False,
        device=0 if key[1] == "cuda" else "cpu",
    )[0]
    xyxy = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4))
    scores = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.zeros(0)
    if len(xyxy):
        heights = xyxy[:, 3] - xyxy[:, 1]
        keep = heights >= keep_relative_height * heights.max()
        xyxy, scores, heights = xyxy[keep], scores[keep], heights[keep]
        order = np.argsort(-heights)
        xyxy, scores = xyxy[order], scores[order]
    return People(xyxy[:, [1, 0, 3, 2]].astype(np.float64), scores.tolist())
