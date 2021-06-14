from numbers import Number

from typing import Any

from .dispatch import dispatch
from .types import Array


@dispatch
def dtype(x: Number):
    return type(x)


@dispatch
def dtype(x: Array):  # noqa: F811, E0102
    return x.dtype


@dispatch
def is_scalar(_: Any):
    return False


@dispatch
def is_scalar(_: Number):  # noqa: F811, E0102
    return True


@dispatch
def is_scalar(x: Array):  # noqa: F811, E0102
    return x.ndim == 0
