"""Reading the numbers people wear: race bibs, jersey numbers.

``read_numbers`` is EasyOCR, digits only, on the chest of each person -- boxes
from ``tnia_ops.detect.detect_people``, or any person detector.

``read_bibs`` is RapidOCR (PaddleOCR's models, on onnxruntime) on boxes known
to hold a number: bibs from ``tnia_ops.detect.detect_bibs``, or chest bands.
On race photos it reads far more than EasyOCR -- 26 of 71 hard bibs against 5
-- with fewer wrong reads.
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


@op(env="photo")
def read_bibs(
    image: RGB,
    boxes: BoxesOf[np.ndarray],
    min_confidence: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.5,
    max_digits: Annotated[int, {"min": 1, "max": 8}] = 4,
    height: Annotated[int, {"min": 32, "max": 2000}] = 160,
    whole: bool = True,
    sure: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.98,
) -> Numbers:
    """Read the number in each box: a bib, or a chest band to search.

    Each box is enlarged to at least *height* pixels and read with RapidOCR,
    which finds the text in it and reads that; with *whole*, the box is also
    read as one line of text, without finding -- better when the box is a
    bib, since the finder can catch only part of one ("5" of an 875 it then
    reads whole at 0.999). The most confident run of digits is kept.

    Args:
        image: RGB uint8.
        boxes: Where to read; one result per box.
        min_confidence: Reads below this are reported with text "".
        max_digits: Longer reads are not numbers people wear.
        height: Boxes smaller than this, in pixels, are enlarged to it.
        whole: The boxes are bibs: also read each whole box as one line.
            Not for a box with more than a bib in it, like a chest band.
        sure: A whole-box read this confident is kept without also finding
            the text in the box -- the slow part.
    """
    import re

    import cv2

    if "rapidocr" not in _models:
        from rapidocr_onnxruntime import RapidOCR

        progress("Loading RapidOCR")
        _models["rapidocr"] = RapidOCR()
    reader = _models["rapidocr"]

    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    H, W = image.shape[:2]
    texts, confidences, found = [], [], np.full((len(boxes), 4), np.nan)
    for i, (y0, x0, y1, x1) in enumerate(boxes):
        if cancel_requested():
            raise RuntimeError(f"Cancelled after {i} of {len(boxes)} boxes")
        progress(f"Reading box {i + 1} of {len(boxes)}", i, len(boxes))
        t, l = max(int(y0), 0), max(int(x0), 0)
        crop = image[t : min(int(np.ceil(y1)), H), l : min(int(np.ceil(x1)), W)]
        best_text, best_conf = "", 0.0
        if crop.shape[0] >= 4 and crop.shape[1] >= 4:
            s = max(1.0, height / crop.shape[0])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            bgr = np.ascontiguousarray(crop[:, :, ::-1])  # RapidOCR takes BGR
            reads = []
            if whole:
                line, _ = reader(bgr, use_det=False, use_cls=False)
                h, w = crop.shape[0] / s, crop.shape[1] / s
                reads += [(text, c, np.array([[0, 0], [w, 0], [w, h], [0, h]])) for text, c in line or []]
            for pass_ in ("whole", "find"):
                if pass_ == "find":
                    if best_conf >= sure:  # finding text is most of the time: only when the whole read is unsure
                        break
                    result, _ = reader(bgr)
                    reads = [(text, c, np.array(quad, dtype=float) / s) for quad, text, c in result or []]
                for text, c, q in reads:
                    runs = re.findall(r"\d+", text)
                    digits = "".join(runs) if re.fullmatch(r"[\d\s]+", text) else max(runs, key=len, default="")
                    if 1 <= len(digits) <= max_digits and float(c) > best_conf:
                        found[i] = [t + q[:, 1].min(), l + q[:, 0].min(), t + q[:, 1].max(), l + q[:, 0].max()]
                        best_text, best_conf = digits, float(c)
        texts.append(best_text if best_conf >= min_confidence else "")
        confidences.append(best_conf)
    return Numbers(texts, confidences, found)
