"""Every op reads as a valid spec. No environment needed."""

import importlib
import pkgutil

import pytest
from opspec.op import OpSpec, is_op

import tnia_ops


def all_ops():
    for info in pkgutil.iter_modules(tnia_ops.__path__):
        module = importlib.import_module(f"tnia_ops.{info.name}")
        for name, value in vars(module).items():
            if is_op(value) and value.__module__ == module.__name__:
                yield f"{info.name}:{name}", value


OPS = dict(all_ops())


def test_there_are_ops():
    assert len(OPS) == 15


@pytest.mark.parametrize("name", sorted(OPS))
def test_each_op_reads_as_a_spec(name):
    spec = OpSpec.from_op(OPS[name])
    assert spec.env in ("skimage", "cupy", "numpy_1x", None)
    assert spec.outputs
    assert OpSpec.from_dict(spec.to_dict()).name == spec.name
