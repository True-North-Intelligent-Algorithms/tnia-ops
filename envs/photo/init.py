# Run by oprunner at worker start-up, on the worker's main thread, after its
# own init script (see oprunner Runner._init_script).
#
# Load the native libraries the photo ops need here, not inside an op. Appose
# runs each op call on a thread of its own, and on Windows a DLL loaded for
# the first time from such a thread can deadlock against the threads torch
# and CUDA have already started: detect_bibs hung, idle, in scipy.linalg's
# BLAS import (transformers -> scipy.optimize -> scipy.linalg), forever. The
# same reason oprunner's init imports numpy first.
import torch  # noqa: F401
import scipy.linalg  # noqa: F401
import scipy.optimize  # noqa: F401
