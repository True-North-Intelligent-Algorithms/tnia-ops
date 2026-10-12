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

Environments are in `envs/<id>/pixi.toml`: `skimage` (CPU), `cupy` (GPU), and
`numpy_1x`, for the simulation ops, whose raster-geometry needs NumPy 1.

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

The notebook's kernel is *tnia-ops (examples)*: Jupyter, scikit-ops' runner,
and tnia-ops, both editable from sibling checkouts. With pixi:

```sh
cd pixi/examples
pixi install
pixi run register-kernel     # once; then pick the kernel in Jupyter or VS Code
pixi run lab
```

Or with uv, from the top folder (the `examples` group has the same things):

```sh
uv sync
uv run python -m ipykernel install --user --name=tnia-ops-examples --display-name 'tnia-ops (examples)'
```
