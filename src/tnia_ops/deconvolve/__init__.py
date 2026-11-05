"""Deconvolution: Richardson-Lucy, on the CPU and the GPU.

One op per backend, because ``@op(env=...)`` is fixed per function and the
backends need different environments.
"""

from .richardson_lucy import richardson_lucy
from .richardson_lucy_cupy import richardson_lucy_cupy

__all__ = ["richardson_lucy", "richardson_lucy_cupy"]
