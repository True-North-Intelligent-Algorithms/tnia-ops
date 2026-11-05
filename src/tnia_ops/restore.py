"""Image restoration with pretrained networks: motion deblur, 2x upscale.

Both are image-to-image models loaded with spandrel, so any architecture it
knows works, from a .pth or .safetensors file. They run on overlapping tiles,
in half precision on a GPU, so a full-size photo fits in memory.

A loaded model is kept for the life of the process: in a warm worker, or a
program calling the ops directly, only the first call pays for loading.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
from opspec.op import Axes, op
from opspec.runtime import cancel_requested, progress
from opspec.types import ImageOf

from ._torch import device

RGB = Annotated[ImageOf[np.ndarray], Axes("y", "x", "c")]

_models: dict = {}


def _load(weights: Path):
    """A spandrel model descriptor from *weights*, on the GPU if there is one,
    cached. ``.model`` is the network, ``.scale`` its upscale factor."""
    key = (str(Path(weights).resolve()), device())
    if key not in _models:
        import spandrel
        import torch

        if not _models.get("extra arches"):  # install() raises the second time
            try:  # Restormer and friends live in the extra architectures
                import spandrel_extra_arches

                spandrel_extra_arches.install()
            except ImportError:
                pass
            _models["extra arches"] = True
        progress(f"Loading {Path(weights).name}")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        for wrapper in ("params_ema", "params"):  # BasicSR-style checkpoints
            if isinstance(state, dict) and wrapper in state:
                state = state[wrapper]
                break
        descriptor = spandrel.ModelLoader().load_from_state_dict(state)
        descriptor.model.eval().to(key[1])
        _models[key] = descriptor
    return _models[key]


def _run_tiled(model, rgb, scale, tile=512, pad=32, what="Restoring"):
    """Run an image-to-image model over a uint8 RGB image in overlapping tiles."""
    import torch

    where = device()
    img = np.ascontiguousarray(rgb).astype(np.float32) / 255
    H, W = img.shape[:2]
    out = np.zeros((H * scale, W * scale, 3), np.float32)
    t = torch.from_numpy(img).permute(2, 0, 1)[None].to(where)
    origins = [(y, x) for y in range(0, H, tile) for x in range(0, W, tile)]
    for i, (y, x) in enumerate(origins):
        if cancel_requested():
            raise RuntimeError(f"Cancelled after {i} of {len(origins)} tiles")
        progress(f"{what}: tile {i + 1} of {len(origins)}", i, len(origins))
        ya, xa = max(y - pad, 0), max(x - pad, 0)
        yb, xb = min(y + tile + pad, H), min(x + tile + pad, W)
        patch = t[:, :, ya:yb, xa:xb]
        ph, pw = patch.shape[2:]
        # Restormer needs sides divisible by 8
        patch = torch.nn.functional.pad(
            patch, (0, (-pw) % 8, 0, (-ph) % 8), mode="reflect"
        )
        with (
            torch.no_grad(),
            torch.autocast(where, dtype=torch.float16, enabled=where == "cuda"),
        ):
            r = model(patch).float().clamp(0, 1)
        r = r[0, :, : ph * scale, : pw * scale].permute(1, 2, 0).cpu().numpy()
        yy, xx = (y - ya) * scale, (x - xa) * scale
        hh, ww = (min(y + tile, H) - y) * scale, (min(x + tile, W) - x) * scale
        out[y * scale : y * scale + hh, x * scale : x * scale + ww] = r[
            yy : yy + hh, xx : xx + ww
        ]
    return (out * 255).round().astype(np.uint8)


@op(env="photo")
def motion_deblur(
    image: RGB,
    weights: Path,
    tile: Annotated[int, {"min": 64, "max": 2048}] = 512,
) -> ImageOf[np.ndarray]:
    """Remove motion blur with a pretrained network.

    Made for Restormer's motion-deblurring checkpoint
    (https://github.com/swz30/Restormer/releases, motion_deblurring.pth);
    any same-size spandrel model works.

    Args:
        image: RGB uint8.
        weights: The model file.
        tile: Tile side, in pixels; smaller uses less GPU memory.
    """
    return _run_tiled(_load(weights).model, image, 1, tile, what="Deblurring")


@op(env="photo")
def upscale(
    image: RGB,
    weights: Path,
    tile: Annotated[int, {"min": 64, "max": 2048}] = 512,
) -> ImageOf[np.ndarray]:
    """Upscale with a pretrained super-resolution network.

    Made for Real-ESRGAN x2plus
    (https://github.com/xinntao/Real-ESRGAN/releases, RealESRGAN_x2plus.pth);
    the scale is whatever the model's is.

    Args:
        image: RGB uint8.
        weights: The model file.
        tile: Tile side, in input pixels; smaller uses less GPU memory.
    """
    model = _load(weights)
    return _run_tiled(model.model, image, model.scale, tile, what="Upscaling")
