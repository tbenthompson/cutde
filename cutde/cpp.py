import logging

import numpy as np

import cutde.cpp_backend_double
import cutde.cpp_backend_float

logger = logging.getLogger(__name__)


def to(arr, float_type):
    return arr.ravel().astype(float_type)


def zeros(shape, float_type):
    return np.zeros(shape, dtype=float_type)


def empty(shape, float_type):
    return np.empty(shape, dtype=float_type)


def get(arr):
    return arr


def max_block_size(requested):
    return 1


def load_module(
    tmpl_name, tmpl_dir=None, save_code=False, no_caching=False, tmpl_args=None
):
    if tmpl_args["float_type"] == "float":
        return cutde.cpp_backend_float
    else:
        return cutde.cpp_backend_double
