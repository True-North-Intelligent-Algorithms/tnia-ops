"""tnia-python's utilities, as ops.

Ops are grouped into namespaces by what they do, laid out like scikit-ops':
a namespace scikit-ops has as a folder (``deconvolve``, ``detect``,
``kernels``, ``segment``) is a folder here too, one module per op, even if
there is only one; any other is a single module (``morphology.py``). A
folder's ``__init__.py`` re-exports its ops, so either way they are imported
from the namespace::

    from tnia_ops.deconvolve import richardson_lucy
    from tnia_ops.morphology import fill_holes

Heavy imports, tnia-python's included, stay inside the op functions, so
reading these ops needs only opspec and numpy.
"""

__version__ = "0.0.1"
