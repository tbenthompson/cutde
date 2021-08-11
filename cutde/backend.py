# noqa: F401

import logging
import os

import numpy as np

try:
    if os.environ.get("CUTDE_USE_BACKEND", "cuda") != "cuda":
        # The CUTDE_USE_BACKEND environment variable overrides the normal
        # choice of backend.
        # This can be helpful for testing purposes when it might be nice to run
        # with OpenCL or C++ even if CUDA is installed.
        raise ImportError

    from cutde.cuda import (  # noqa: F401
        empty,
        get,
        load_module,
        max_block_size,
        to,
        zeros,
    )

    which_backend = "cuda"
except ImportError:
    try:
        if os.environ.get("CUTDE_USE_BACKEND", "opencl") != "opencl":
            raise ImportError
        from cutde.opencl import (  # noqa: F401
            empty,
            get,
            load_module,
            max_block_size,
            to,
            zeros,
        )

        which_backend = "opencl"

    except ImportError:
        from cutde.cpp import (  # noqa: F401
            empty,
            get,
            load_module,
            max_block_size,
            to,
            zeros,
        )

        which_backend = "cpp"

logger = logging.getLogger(__name__)
logger.debug(f'cutde is using the "{which_backend}" backend')


def np_to_c_type(t):
    if t == np.float32:
        return "float"
    elif t == np.float64:
        return "double"


def intervals(length, step_size):
    out = []
    next_start = 0
    next_end = step_size
    while next_end < length + step_size:
        this_end = min(next_end, length)
        out.append((next_start, this_end))
        next_start += step_size
        next_end += step_size
    return out
