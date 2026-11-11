# tnia-ops

[tnia-python](https://github.com/True-North-Intelligent-Algorithms/tnia-python)'s
utilities, as ops: plain Python functions declared with
[opspec](https://github.com/apposed/scikit-ops/tree/main/opspec), so a
runner can run them in their own environments and a front end can show them.

Each op wraps a tnia-python function and changes nothing about it.

| Namespace | Ops |
|---|---|
| `background` | `rolling_ball`, `dilate_subtract` |
| `deconvolve` | `richardson_lucy`, `richardson_lucy_cupy` |
| `kernels` | `gaussian_psf`, `paraxial_psf` |
| `morphology` | `fill_holes`, `connect_2d_in_3d` |
| `segment` | `local_global_threshold`, `find_large_objects`, `object_boxes`, `threshold_and_fill` (a workflow) |
| `simulation` | `random_circles`, `random_spheres`, `grid_of_circles` |
| `photo` | `relight`, `color_correct`, `unsharp`, `denoise`, `sharpness`, `subject_crop` |
| `restore` | `motion_deblur`, `upscale` (pretrained networks, via spandrel) |
| `detect` | `detect_people` (YOLO, person class), `detect_bibs` (OWLv2, zero-shot), `detect_poses` (YOLO pose keypoints) |
| `ocr` | `read_numbers` (EasyOCR on chests), `read_bibs` (RapidOCR on given boxes) |

Namespaces are laid out like scikit-ops': one scikit-ops has as a folder
(`deconvolve`, `detect`, `kernels`, `segment`) is a folder here too, one file
per op, even if there is only one; the rest are single files. Either way, ops
are imported from the namespace: `from tnia_ops.deconvolve import richardson_lucy`.

Environments are in `envs/<id>/pixi.toml`: `skimage` (CPU), `cupy` (GPU), and
`numpy_1x`, for the simulation ops, whose raster-geometry needs NumPy 1, and
`photo` (OpenCV and PyTorch) for the photo, restore, detect and ocr ops.

The photo ops are not wrappers of tnia-python: they came from a race-photo
app, which calls them directly. Model weights are parameters (`weights: Path`),
never downloaded by the op -- except `detect_bibs`, whose `model` is a Hugging
Face id that transformers downloads and caches on first use.

Some names are the same as scikit-ops' on purpose -- ops called
`richardson_lucy` and `connect_2d_in_3d`, environments called `skimage` and
`cupy` -- to find out how a runner copes with two collections that clash.

```sh
uv sync
uv run pytest      # every op reads as a valid spec; no environment needed
```

## Examples

tnia-ops has no runner of its own. `examples/hybrid.ipynb` runs it with
scikit-ops' runner, in one pipeline with a scikit-ops op, and shows which
environment each op runs in and where it is built.

The notebook's kernel is *tnia-ops (editable)*: Jupyter, scikit-ops' runner,
and tnia-ops, both editable from sibling checkouts. With pixi:

```sh
cd pixi/tnia-ops-editable
pixi install
pixi run register-kernel     # once; then pick the kernel in Jupyter or VS Code
pixi run lab
```

Or with uv, from the top folder (the `examples` group has the same things):

```sh
uv sync
uv run python -m ipykernel install --user --name=tnia-ops-examples --display-name 'tnia-ops (examples)'
```
