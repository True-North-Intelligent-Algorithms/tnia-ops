"""Every op reads as a valid spec. No environment needed."""

import importlib
import pkgutil

import pytest
from opspec.op import OpSpec, is_op

import tnia_ops


def all_ops():
    for info in pkgutil.walk_packages(tnia_ops.__path__, "tnia_ops."):
        if info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        module = importlib.import_module(info.name)
        for name, value in vars(module).items():
            if is_op(value) and value.__module__ == module.__name__:
                yield f"{info.name}:{name}", value


OPS = dict(all_ops())


def test_there_are_ops():
    assert len(OPS) == 28


@pytest.mark.parametrize("name", sorted(OPS))
def test_each_op_reads_as_a_spec(name):
    spec = OpSpec.from_op(OPS[name])
    assert spec.env in ("skimage", "cupy", "numpy_1x", "photo", None)
    assert spec.outputs
    assert OpSpec.from_dict(spec.to_dict()).name == spec.name
