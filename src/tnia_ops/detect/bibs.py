"""Finding race bibs, zero-shot: no bib training data needed.

``detect_bibs`` is OWLv2, an open-vocabulary detector: it is told in words
what to look for ("a race bib number") and finds it. Not as precise as a
detector trained on bibs -- it also boxes torsos and signs, so read the boxes
and keep those with a number -- but it finds bibs a text detector misses:
single big digits, crumpled ones, ones half behind a hand.

OWLv2 sees the image at 960 pixels, where a bib far back in a 24-megapixel
photo is a few pixels wide. Given *regions* -- the people in the photo --
it looks in each one at full size instead.
"""

from typing import Annotated, NamedTuple

import numpy as np
from opspec.op import Axes, op
from opspec.runtime import cancel_requested, progress
from opspec.types import BoxesOf, ImageOf

from .._torch import device

RGB = Annotated[ImageOf[np.ndarray], Axes("y", "x", "c")]

_models: dict = {}


class Bibs(NamedTuple):
    #: (N, 4) as [min_y, min_x, max_y, max_x], most confident first.
    boxes: BoxesOf[np.ndarray]
    #: One detection score per box (OWLv2's are low: 0.15 is a fair bib).
    scores: list[float]
    #: Which region each box was found in; -1 when no regions were given.
    regions: list[int]


@op(env="photo")
def detect_bibs(
    image: RGB,
    regions: BoxesOf[np.ndarray] | None = None,
    prompt: str = "a race bib number",
    threshold: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.15,
    overlap: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.5,
    model: str = "google/owlv2-base-patch16-ensemble",
) -> Bibs:
    """Find the race bibs in a photo.

    Args:
        image: RGB uint8.
        regions: Where to look, e.g. the people from detect_people; each is
            searched at full size. None searches the whole photo at once.
        prompt: What a bib is, in words.
        threshold: Minimum detection score.
        overlap: Boxes overlapping a better one by more than this (IoU) are
            dropped.
        model: An OWLv2 checkpoint: a Hugging Face model id (downloaded and
            cached on first use) or a local folder.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Owlv2ForObjectDetection

    where = device()
    key = (model, where)
    if key not in _models:
        progress(f"Loading {model}")
        _models[key] = (
            AutoProcessor.from_pretrained(model),
            Owlv2ForObjectDetection.from_pretrained(model).to(where).eval(),
        )
    proc, net = _models[key]

    H, W = image.shape[:2]
    if regions is None:
        windows = [(0, 0, H, W)]
    else:
        windows = [
            (max(int(t), 0), max(int(l), 0), min(int(np.ceil(b)), H), min(int(np.ceil(r)), W))
            for t, l, b, r in np.asarray(regions, dtype=float).reshape(-1, 4)
        ]
    boxes, scores, found_in = [], [], []
    for i, (t, l, b, r) in enumerate(windows):
        if cancel_requested():
            raise RuntimeError(f"Cancelled after {i} of {len(windows)} regions")
        progress(f"Finding bibs: region {i + 1} of {len(windows)}", i, len(windows))
        if b - t < 8 or r - l < 8:
            continue
        crop = Image.fromarray(np.ascontiguousarray(image[t:b, l:r]))
        crop.thumbnail((1440, 1440))  # OWLv2 sees 960 anyway; preparing 24 megapixels is the slow part
        inputs = proc(text=[[prompt]], images=crop, return_tensors="pt").to(where)
        with torch.no_grad():
            out = net(**inputs)
        side = max(b - t, r - l)  # OWLv2 pads to a square
        res = proc.post_process_grounded_object_detection(
            outputs=out, threshold=threshold, target_sizes=torch.tensor([[side, side]], device=where)
        )[0]
        for (x0, y0, x1, y1), s in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()):
            y0, y1 = np.clip([y0, y1], 0, b - t) + t
            x0, x1 = np.clip([x0, x1], 0, r - l) + l
            if y1 > y0 and x1 > x0:
                boxes.append([y0, x0, y1, x1])
                scores.append(float(s))
                found_in.append(i if regions is not None else -1)

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    keep = _nms(boxes, np.asarray(scores), overlap)
    return Bibs(boxes[keep], [scores[k] for k in keep], [found_in[k] for k in keep])


def _nms(boxes, scores, overlap):
    """Indices of the boxes to keep, best first: drop any overlapping a kept one by more than *overlap*."""
    keep = []
    for i in np.argsort(-scores) if len(scores) else []:
        t, l, b, r = boxes[i]
        ok = True
        for j in keep:
            T, L, B, R = boxes[j]
            inter = max(0.0, min(b, B) - max(t, T)) * max(0.0, min(r, R) - max(l, L))
            union = (b - t) * (r - l) + (B - T) * (R - L) - inter
            if union > 0 and inter / union > overlap:
                ok = False
                break
        if ok:
            keep.append(int(i))
    return keep
