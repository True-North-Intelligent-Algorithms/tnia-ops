"""Watch a tnia-ops op report progress through scikit-ops' runner.

    uv run python examples/progress_demo.py
"""

import numpy as np
import oprunner

from tnia_ops.background import rolling_ball


def show(event):
    if getattr(event, "message", None):
        print(event.message)


stack = (np.random.default_rng(0).random((10, 256, 256)) * 100).astype(np.float32)
with oprunner.Runner() as runner:
    runner.run(rolling_ball, image=stack, radius=25, smooth=False, on_progress=show)
print("done")
