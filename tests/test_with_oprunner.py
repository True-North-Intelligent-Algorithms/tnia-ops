"""tnia-ops run by scikit-ops' runner: progress and cancel, through opspec.

tnia-ops itself depends on opspec alone; these need scikit-ops (the examples
group), and build tnia's skimage environment the first time.
"""

import numpy as np
import pytest

oprunner = pytest.importorskip("oprunner")

from tnia_ops.background import rolling_ball
from tnia_ops.segment import object_boxes, threshold_and_fill


def stack(planes, size):
    rng = np.random.default_rng(0)
    return (rng.random((planes, size, size)) * 100).astype(np.float32)


def messages(events):
    return [e.message for e in events if getattr(e, "message", None)]


def test_progress_reaches_the_caller():
    events = []
    with oprunner.Runner() as runner:
        result = runner.run(
            rolling_ball,
            image=stack(4, 64),
            radius=10,
            smooth=False,
            on_progress=events.append,
        )
    assert result.corrected.shape == (4, 64, 64)
    planes = [m for m in messages(events) if m.startswith("Plane")]
    assert planes == [f"Plane {i} of 4" for i in range(1, 5)]


def test_cancel_stops_between_planes():
    tasks = []

    def cancel_after_the_first_plane(event):
        if getattr(event, "message", None) == "Plane 1 of 30":
            tasks[0].cancel()

    with oprunner.Runner() as runner, pytest.raises(Exception, match="(?i)cancel"):
        runner.run(
            rolling_ball,
            image=stack(30, 256),
            radius=25,
            smooth=False,
            on_progress=cancel_after_the_first_plane,
            on_start=tasks.append,
        )


def test_an_op_uses_ophelper_in_its_environment():
    labels = np.zeros((32, 32), np.int32)
    labels[2:5, 3:9] = 1
    labels[10:20, 10:12] = 2
    with oprunner.Runner() as runner:
        boxes = runner.run(object_boxes, labels=labels)
    assert boxes.tolist() == [[2, 3, 5, 9], [10, 10, 20, 12]]


def test_a_workflow_runs_its_steps_through_the_runner():
    image = np.zeros((64, 64), np.float32)
    image[10:30, 10:30] = 100
    image[19:21, 19:21] = 0  # a hole
    events = []
    with oprunner.Runner() as runner:
        labels = runner.run(
            threshold_and_fill,
            image=image,
            window=61,
            global_threshold=50,
            on_progress=events.append,
        )
        # The steps ran in tnia's environment, so a worker was started.
        assert any(key[0] == "skimage" for key in runner._services)
    assert "Thresholding" in messages(events)
    assert "Filling holes" in messages(events)
    assert labels[20, 20] == labels[12, 12] != 0  # the hole is filled
