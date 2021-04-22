from numbers import Number

from typing import Any, Union
from plum import dispatch

import numpy as np
import jax
import jaxlib

ArrayT = Union[np.ndarray, jaxlib.xla_extension.DeviceArray, jax.core.Tracer]


@dispatch
def dtype(x: Number):
    return type(x)


@dispatch
def dtype(x: ArrayT):
    return x.dtype


@dispatch
def is_scalar(x: Any):
    return False


@dispatch
def is_scalar(x: Number):
    return True


@dispatch
def is_scalar(x: ArrayT):
    return x.ndim == 0
