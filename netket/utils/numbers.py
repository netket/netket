from numbers import Number

from typing import Any, Union

import numpy as np
import jax
import jaxlib

from .dispatch import dispatch
from .types import Array


@dispatch
def dtype(x: Number):
    return type(x)


@dispatch
def dtype(x: Array):
    return x.dtype


@dispatch
def is_scalar(x: Any):
    return False


@dispatch
def is_scalar(x: Number):
    return True


@dispatch
def is_scalar(x: Array):
    return x.ndim == 0
