"""Background subtraction."""

from typing import Annotated, NamedTuple

import numpy as np
from opspec.op import op
from opspec.types import ImageOf


class Corrected(NamedTuple):
    #: The image, background subtracted.
    corrected: ImageOf[np.ndarray]
    #: The background that was subtracted.
    background: ImageOf[np.ndarray]


@op(env="skimage")
def rolling_ball(
    image: ImageOf[np.ndarray],
    radius: Annotated[float, {"min": 1.0, "max": 500.0}] = 50.0,
    smooth: bool = True,
    downsample: Annotated[int, {"min": 1, "max": 16}] = 1,
) -> Corrected:
    """Subtract a rolling-ball background, as ImageJ's Subtract Background.

    Args:
        image: The image. Integer (uint8, uint16) if smooth is on.
        radius: The ball's radius, in pixels.
        smooth: Whether to smooth before estimating the background.
        downsample: Estimate the background at 1/n size, for speed.
    """
    from tnia.background.rolling_ball import rolling_ball_ds

    background = rolling_ball_ds(
        image, radius, perform_mean=smooth, downsample=downsample
    )
    return Corrected(image - background, background)


@op(env="skimage")
def dilate_subtract(
    image: ImageOf[np.ndarray],
    percentile: Annotated[float, {"min": 0.0, "max": 100.0}] = 50.0,
) -> Corrected:
    """Subtract a background estimated by dilating the image.

    Args:
        image: The image.
        percentile: Sets the seed's height; higher keeps less.
    """
    from tnia.background.dilate_subtract import dilate_subtract as tnia_dilate_subtract

    corrected, background, _, _ = tnia_dilate_subtract(image, percentile)
    return Corrected(corrected, background)
