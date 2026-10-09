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
| `segment` | `local_global_threshold`, `find_large_objects` |
| `simulation` | `random_circles`, `grid_of_circles` |

Environments are in `envs/<id>/pixi.toml`: `skimage` (CPU), `cupy` (GPU), and
`numpy_1x`, for the simulation ops, whose raster-geometry needs NumPy 1.

Some names are the same as scikit-ops' on purpose -- ops called
`richardson_lucy` and `connect_2d_in_3d`, environments called `skimage` and
`cupy` -- to find out how a runner copes with two collections that clash.

```sh
uv sync
uv run pytest      # every op reads as a valid spec; no environment needed
```
