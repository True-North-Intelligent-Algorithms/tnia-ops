"""Reading the numbers people wear: race bibs, jersey numbers.

``read_numbers`` is EasyOCR, digits only, on the chest of each person -- boxes
from ``tnia_ops.detect.detect_people``, or any person detector.
"""

from typing import Annotated, NamedTuple

import numpy as np
from opspec.op import Axes, op
from opspec.runtime import cancel_requested, progress
from opspec.types import BoxesOf, ImageOf

from ._torch import device

RGB = Annotated[ImageOf[np.ndarray], Axes("y", "x", "c")]

_models: dict = {}


class Numbers(NamedTuple):
    #: The number read for each person; "" where none was read confidently.
    texts: list[str]
    #: OCR confidence of the best read per person, even an unconfident one.
    confidences: list[float]
    #: (N, 4) where that read was, [min_y, min_x, max_y, max_x]; NaN for none.
    boxes: BoxesOf[np.ndarray]


@op(env="photo")
def read_numbers(
    image: RGB,
    people: BoxesOf[np.ndarray],
    min_confidence: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.5,
    max_digits: Annotated[int, {"min": 1, "max": 8}] = 4,
    chest_top: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.15,
    chest_bottom: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.70,
) -> Numbers:
    """Read the number each person wears: a race bib, a jersey.

    Looks at the chest -- a band of each person's box -- enlarged so the
    digits are big enough to read, and keeps the most confident read.

    Args:
        image: RGB uint8.
        people: One box per person.
        min_confidence: Reads below this are reported with text "".
        max_digits: Longer reads are not numbers people wear.
        chest_top: Where the chest starts, as a fraction of the box height.
        chest_bottom: Where it ends.
    """
    import re

    import cv2

    key = ("easyocr", device())
    if key not in _models:
        import easyocr

        progress("Loading EasyOCR")
        _models[key] = easyocr.Reader(["en"], gpu=key[1] == "cuda", verbose=False)
    reader = _models[key]

    boxes = np.asarray(people, dtype=float).reshape(-1, 4)
    pattern = re.compile(rf"\d{{1,{max_digits}}}")
    texts, confidences, found = [], [], np.full((len(boxes), 4), np.nan)
    for i, (y0, x0, y1, x1) in enumerate(boxes.astype(int)):
        if cancel_requested():
            raise RuntimeError(f"Cancelled after {i} of {len(boxes)} people")
        progress(f"Reading person {i + 1} of {len(boxes)}", i, len(boxes))
        h = y1 - y0
        top = y0 + int(chest_top * h)
        chest = image[top : y0 + int(chest_bottom * h), x0:x1]
        best_text, best_conf = "", 0.0
        if chest.size:
            s = max(1.0, 500 / chest.shape[0])
            chest = cv2.resize(chest, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            for quad, text, c in reader.readtext(chest, allowlist="0123456789"):
                if pattern.fullmatch(text) and c > best_conf:
                    q = np.array(quad, dtype=float) / s
                    found[i] = [
                        top + q[:, 1].min(),
                        x0 + q[:, 0].min(),
                        top + q[:, 1].max(),
                        x0 + q[:, 0].max(),
                    ]
                    best_text, best_conf = text, float(c)
        texts.append(best_text if best_conf >= min_confidence else "")
        confidences.append(best_conf)
    return Numbers(texts, confidences, found)
