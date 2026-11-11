"""Finding the people in a photo.

``detect_people`` is a COCO YOLO checkpoint kept to its person class, at a
chosen inference size. scikit-ops' ``yolo`` runs any checkpoint, but at the
size it was trained at (640 for COCO's), which loses people far from the
camera in a 4000-pixel photo; this runs the whole photo at 1280 instead.

``detect_poses`` is a COCO YOLO pose checkpoint: each person's 17 keypoints
(nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) -- e.g. to
tell whether an arm crosses something in front of the body.
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


#: COCO's keypoint order, as detect_poses returns them.
KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)  # fmt: skip


class Poses(NamedTuple):
    #: (N, 4) as [min_y, min_x, max_y, max_x], one per person.
    boxes: BoxesOf[np.ndarray]
    #: (N, 17, 3): y, x and confidence of each keypoint, in KEYPOINTS order.
    keypoints: np.ndarray


@op(env="photo")
def detect_poses(
    image: RGB,
    weights: Path,
    size: Annotated[int, {"min": 320, "max": 4096}] = 1280,
    conf: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.25,
) -> Poses:
    """Find each person's body keypoints: shoulders, elbows, wrists and so on.

    Args:
        image: RGB uint8.
        weights: A COCO-trained YOLO pose checkpoint, e.g. yolo11m-pose.pt
            (https://github.com/ultralytics/assets/releases).
        size: Inference size, in pixels, of the photo's long side.
        conf: Minimum person confidence.
    """
    from ultralytics import YOLO

    key = (str(Path(weights).resolve()), device())
    if key not in _models:
        progress(f"Loading {Path(weights).name}")
        _models[key] = YOLO(str(weights))
    progress("Finding poses")
    result = _models[key].predict(
        np.ascontiguousarray(image[:, :, ::-1]),  # ultralytics takes BGR arrays
        imgsz=size,
        conf=conf,
        verbose=False,
        device=0 if key[1] == "cuda" else "cpu",
    )[0]
    if not len(result.boxes):
        return Poses(np.zeros((0, 4)), np.zeros((0, 17, 3)))
    xyxy = result.boxes.xyxy.cpu().numpy()
    kp = result.keypoints.data.cpu().numpy()  # x, y, confidence
    return Poses(xyxy[:, [1, 0, 3, 2]].astype(np.float64), kp[:, :, [1, 0, 2]].astype(np.float64))
