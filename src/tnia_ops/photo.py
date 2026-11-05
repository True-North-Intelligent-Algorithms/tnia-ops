"""Photo clean-up: light, color, sharpness, and cropping to the subject.

Written for outdoor photos of people -- race photos, first -- but nothing
here knows about races. Images are RGB uint8, (y, x, c). Boxes are
[min_y, min_x, max_y, max_x], as everywhere in opspec.
"""

from typing import Annotated, NamedTuple

import numpy as np
from opspec.op import Axes, op
from opspec.types import BoxesOf, ImageOf

RGB = Annotated[ImageOf[np.ndarray], Axes("y", "x", "c")]


class Lit(NamedTuple):
    #: The image, relit.
    image: ImageOf[np.ndarray]
    #: Whether the subject was dark, and so lifted.
    brightened: bool
    #: Whether the subject was against bright light, and so given local contrast.
    backlit: bool


@op(env="photo")
def relight(
    image: RGB,
    subject: BoxesOf[np.ndarray] | None = None,
    dark_below: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.40,
    target_brightness: Annotated[float, {"min": 0.1, "max": 0.9}] = 0.48,
) -> Lit:
    """Lift a dark subject, and tame backlight with local contrast.

    The lift is a gamma on luminance with a soft shoulder, so bright sky and
    sun do not blow out.

    Args:
        image: RGB uint8.
        subject: Boxes around the subject; their union is what is measured.
            None measures the whole image.
        dark_below: A subject darker than this (mean luminance, 0-1) is lifted.
        target_brightness: What a dark subject is lifted towards.
    """
    import cv2

    lab = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0
    boxes = np.empty((0, 4)) if subject is None else np.asarray(subject).reshape(-1, 4)
    if len(boxes):
        y0, x0 = int(max(boxes[:, 0].min(), 0)), int(max(boxes[:, 1].min(), 0))
        y1, x1 = int(boxes[:, 2].max()), int(boxes[:, 3].max())
        region = L[y0:y1, x0:x1]
        subject_mean = float(region.mean()) if region.size else float(L.mean())
    else:
        subject_mean = float(L.mean())
    backlit = float(L.mean()) - subject_mean > 0.08 or float((L > 0.95).mean()) > 0.02
    dark = subject_mean < dark_below
    if dark:
        gamma = np.clip(
            np.log(target_brightness) / np.log(max(subject_mean, 0.05)), 0.55, 1.0
        )
        L = L + (L**gamma - L) * (1 - L) ** 0.5
    if dark or backlit:
        clahe = cv2.createCLAHE(clipLimit=1.6 if backlit else 1.2, tileGridSize=(8, 8))
        L8 = np.clip(L * 255, 0, 255).astype(np.uint8)
        lab[:, :, 0] = cv2.addWeighted(clahe.apply(L8), 0.6, L8, 0.4, 0)
    return Lit(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), bool(dark), bool(backlit))


@op(env="photo")
def color_correct(
    image: RGB,
    green_fix: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.35,
    warmth: Annotated[float, {"min": -10.0, "max": 10.0}] = 2.0,
    saturation: Annotated[float, {"min": 0.5, "max": 1.5}] = 1.08,
) -> ImageOf[np.ndarray]:
    """Pull back a green cast (tree canopy), warm, and restore saturation.

    Args:
        image: RGB uint8.
        green_fix: Fraction of an overall green cast to remove.
        warmth: Added to LAB's b channel; positive is warmer.
        saturation: Multiplies HSV saturation, to give back what brightening took.
    """
    import cv2

    lab = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2LAB).astype(
        np.float32
    )
    a_off = lab[:, :, 1].mean() - 128
    if a_off < 0:
        lab[:, :, 1] -= green_fix * a_off
    lab[:, :, 2] += warmth
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


@op(env="photo")
def unsharp(
    image: RGB,
    amount: Annotated[float, {"min": 0.0, "max": 3.0}] = 0.6,
    sigma: Annotated[float, {"min": 0.3, "max": 10.0}] = 1.4,
) -> ImageOf[np.ndarray]:
    """Unsharp mask, for mildly soft photos.

    Args:
        image: RGB uint8.
        amount: How much of the detail to add back.
        sigma: Blur width the detail is measured against, in pixels.
    """
    import cv2

    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, blur, -amount, 0)


@op(env="photo")
def denoise(
    image: RGB,
    strength: Annotated[float, {"min": 1.0, "max": 20.0}] = 3.0,
) -> ImageOf[np.ndarray]:
    """Non-local means denoising of color, for noise that brightening brought up.

    Args:
        image: RGB uint8.
        strength: Filter strength, for luminance and color alike.
    """
    import cv2

    return cv2.fastNlMeansDenoisingColored(
        np.ascontiguousarray(image), None, strength, strength, 7, 21
    )


@op(env="photo")
def sharpness(
    image: RGB,
    boxes: BoxesOf[np.ndarray],
    height: Annotated[int, {"min": 50, "max": 4000}] = 600,
) -> list[float]:
    """How sharp each boxed region is: variance of the Laplacian.

    Each region is resized to the same height first, so a near subject and
    a far one are measured alike.

    Args:
        image: RGB uint8.
        boxes: The regions to measure.
        height: The height each region is resized to.
    """
    import cv2

    scores = []
    for y0, x0, y1, x1 in np.asarray(boxes).reshape(-1, 4).astype(int):
        region = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        if region.size == 0:
            scores.append(0.0)
            continue
        s = height / region.shape[0]
        region = cv2.resize(region, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        scores.append(float(cv2.Laplacian(region, cv2.CV_64F).var()))
    return scores


@op
def subject_crop(
    shape: tuple[int, int],
    subject: BoxesOf[np.ndarray],
    crop_if_below: Annotated[float, {"min": 0.0, "max": 1.0}] = 0.60,
    subject_fill: Annotated[float, {"min": 0.2, "max": 1.0}] = 0.78,
    min_crop: Annotated[float, {"min": 0.05, "max": 1.0}] = 0.22,
) -> BoxesOf[np.ndarray] | None:
    """A crop around the subject, with the image's aspect ratio, or None.

    None means "keep the whole frame": there is no subject, it already
    fills the frame, or the crop would hardly be smaller.

    Args:
        shape: The image's (height, width).
        subject: Boxes around the subject.
        crop_if_below: Crop only a subject shorter than this fraction of the
            image height.
        subject_fill: How much of the crop's height (or width) the subject takes.
        min_crop: The crop is at least this fraction of the image height.

    Returns:
        crop: One box, (1, 4) as [min_y, min_x, max_y, max_x].
    """
    H, W = shape
    b = np.asarray(subject, dtype=float).reshape(-1, 4)
    if not len(b):
        return None
    y0, x0, y1, x1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    gh, gw = y1 - y0, x1 - x0
    if gh >= crop_if_below * H:
        return None
    aspect = W / H
    ch = min(max(gh / subject_fill, (gw / subject_fill) / aspect, min_crop * H), H)
    cw = min(ch * aspect, W)
    if cw >= 0.9 * W:
        return None
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    left = int(np.clip(cx - cw / 2, 0, W - cw))
    top = int(np.clip(cy - ch / 2, 0, H - ch))
    return np.array([[top, left, top + int(ch), left + int(cw)]])
