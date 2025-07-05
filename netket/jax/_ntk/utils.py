# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General-purpose internal utilities.

If a function or class is used in multiple modules, put it here.
"""

import functools
import inspect
import operator
from typing import Any, TypeVar, Union
from collections.abc import Callable, Iterable, Sequence, Sized
import warnings

import jax
from jax import core
from jax import random
import jax.numpy as jnp
from jax.tree_util import tree_all
from jax.tree_util import tree_map
import numpy as np


PyTree = Any


Axes = Union[int, Sequence[int]]


def is_list_or_tuple(x) -> bool:
    # We do not want to return True if x is a subclass of list or tuple since
    # otherwise this will return true for namedtuples.
    return type(x) == list or type(x) == tuple


def nt_tree_fn(
    nargs: int | None = None,
    tree_structure_argnum: int | None = None,
    reduce: Callable = lambda x: x,
):
    """Convert a function that acts on single inputs to one that acts on trees.

    `nt_tree_fn` treats the first `nargs` arguments as NTTrees and the remaining
    arguments as broadcasted over the tree structure. `nt_tree_fn` then calls the
    function on each leaf of the tree. Each node of the tree optionally calls a
    reduce function over the values of its children.

    If `tree_structure_argnum` is None then each of the NTTrees must have the same
    structure. If `tree_structure_argnum` is an integer then a specific tree is
    used to infer the structure.

    Args:
      nargs:
        The number of arguments to be treated as NTTrees. If `nargs` is `None`
        then all the arguments are used. `nargs` can also be negative which
        follows numpy's semantics for array indexing.

      tree_structure_argnum:
        The argument used to infer the tree structure to be traversed. If
        `tree_structure_argnum` is None then a check is performed to ensure that
        all trees have the same structure.

      reduce:
        A callable that is applied recursively by each internal tree node to its
        children.

    Returns:
      A decorator `tree_fn` that transforms a function, `fn`, from acting on
      leaves to acting on NTTrees.
    """

    def check_tree_structure(args):
        """Ensure the structure of the trees in each of the `nargs` is the same."""
        if any(is_list_or_tuple(x) for x in args):
            if not all(type(x) == type(args[0]) for x in args[1:]):
                raise TypeError(
                    f"Inconsistent NTTree structure found. "
                    f"Node Types: {[type(x) for x in args]}."
                )
            for x in zip(*args):
                # Regarding the use of zip, consider an example `x1 = x2 = (1, (1, 1))`.
                # We would like to determine whether these two trees have the same
                # structure.

                # On the first recurrence `x1` and `x2` are both tuples so the check
                # passes and `zip(*args) = [(1, 1), ((1, 1), (1, 1))]` so that
                # `(check_tree_structure(x) for x in zip(x1, x2))` will first check that
                # the first element of `x1` has the same tree structure as the first
                # element of `x2` and then the second element and so on.
                check_tree_structure(x)

    def tree_fn(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            _nargs = len(args) if nargs is None else nargs
            recurse, norecurse = args[:_nargs], args[_nargs:]

            structure_argnum = tree_structure_argnum
            if structure_argnum is None:
                check_tree_structure(recurse)
                structure_argnum = 0

            if is_list_or_tuple(args[structure_argnum]):
                list_or_tuple = type(args[structure_argnum])
                return reduce(
                    list_or_tuple(
                        wrapped_fn(*(xs + norecurse), **kwargs) for xs in zip(*recurse)
                    )
                )
            return fn(*args, **kwargs)

        return wrapped_fn

    return tree_fn


def all_none(x, attr: str | None = None) -> bool:
    get_fn = (lambda x: x) if attr is None else lambda x: getattr(x, attr)
    return tree_all(tree_map(lambda x: get_fn(x) is None, x))


def wraps(f):
    def wrapper(g):
        @functools.wraps(f)
        def h(*args, **kwargs):
            return g(*args, **kwargs)

        h.__signature__ = inspect.signature(f)
        return h

    return wrapper


@nt_tree_fn(nargs=2, reduce=lambda x: jnp.all(jnp.array(x)))
def x1_is_x2(
    x1: jnp.ndarray, x2: jnp.ndarray | None = None, eps: float = 1e-12
) -> bool | jnp.ndarray:
    if not isinstance(x1, (np.ndarray, jnp.ndarray)):
        raise TypeError(f"`x1` must be an ndarray. A {type(x1)} is found.")

    if x2 is None:
        return True

    if x1 is x2:
        return True

    if x1.shape != x2.shape:
        return False

    if jax.default_backend() == "tpu":
        eps = 1e-4

    try:
        diff = x1 - x2
    except TypeError:
        # inputs are e.g. custom PRNGKeys which don't define subtraction.
        return jnp.all(x1 == x2)
    else:
        return jnp.all(jnp.abs(diff) < eps)


def _get_ndim(x: int | Sized | jnp.ndarray) -> int:
    """Get number of dimensions given number of dimensions / shape / array."""
    if hasattr(x, "ndim"):
        n = x.ndim
    elif hasattr(x, "__len__"):
        n = len(x)
    elif isinstance(x, int):
        n = x
    else:
        raise TypeError(x, type(x))
    return n


def canonicalize_axis(axis: Axes, x: int | Sized | jnp.ndarray) -> list[int]:
    """Converts axis into a sorted non-negative list.

    Args:
      axis: input axis.
      x: array / shape / number of dimensions.

    Returns:
      A sorted list of integer axes.
    """
    axis = [axis] if isinstance(axis, int) else list(axis)
    n = _get_ndim(x)
    return list(set(np.arange(n)[axis]))


def zip_axes(
    x: jnp.ndarray, start_axis: int = 0, end_axis: int | None = None
) -> jnp.ndarray:
    """Zip (interleave) axes starting from `start_axis`.

    Changes the shape as follows:
    `[..., X, Y, Z, ..., X, Y, Z, ...] -> [..., X, X, ..., Y, Y, ..., Z, Z, ...]`

    Args:
      x: `jnp.ndarray` with an even number of dimensions following `start_axis`.
      start_axis: `int`, number of axis from which to zip (interleave).
      end_axis: `int`, number of axis until which to zip (interleave).

    Returns:
      A `jnp.ndarray` with a new shape.
    """
    return _zip_axes(x, start_axis, end_axis, unzip=False)


def unzip_axes(
    x: jnp.ndarray, start_axis: int = 0, end_axis: int | None = None
) -> jnp.ndarray:
    """Unzip (de-interleave) axes starting from `start_axis`.

    Changes the shape as follows:
    `[..., X, X, ..., Y, Y, ..., Z, Z, ...] -> [..., X, Y, Z, ..., X, Y, Z, ...]`

    Args:
      x: `jnp.ndarray` with an even number of dimensions following `start_axis`.
      start_axis: `int`, number of axis from which to unzip (de-interleave).
      end_axis: `int`, number of axis until which to unzip (de-interleave).

    Returns:
      A `jnp.ndarray` with a new shape.
    """
    return _zip_axes(x, start_axis, end_axis, unzip=True)


def _zip_axes(
    x: jnp.ndarray,
    start_axis: int = 0,
    end_axis: int | None = None,
    unzip: bool = False,
) -> jnp.ndarray:
    """Zip/unzip (interleave/de-interleave) axes starting from `start_axis`.

    Changes the shape as follows:
      If `unzip == True`:
      `[..., X, X, ..., Y, Y, ..., Z, Z, ...] -> [..., X, Y, Z, ..., X, Y, Z, ..]`
      If `unzip == False`:
      `[..., X, Y, Z, ..., X, Y, Z, ...] -> [..., X, X, ..., Y, Y, ..., Z, Z, ..]`

    Args:
      x: `jnp.ndarray` with an even number of dimensions following `start_axis`.
      start_axis: `int`, number of axis from which to zip/unzip.
      end_axis: `int`, number of axis until which to zip/unzip.
      unzip: `bool`, set to `True` to unzip instead of zip.

    Returns:
      A `jnp.ndarray` with a new shape.
    """
    if end_axis is None:
        end_axis = x.ndim

    half_ndim, ragged = divmod(end_axis - start_axis, 2)
    if ragged:
        raise ValueError(
            f"Need even number of axes to zip, got {end_axis - start_axis}."
        )

    odd_axes = range(start_axis + 1, end_axis, 2)
    last_axes = range(end_axis - half_ndim, end_axis)

    if unzip:
        x = jnp.moveaxis(x, odd_axes, last_axes)
    else:
        x = jnp.moveaxis(x, last_axes, odd_axes)
    return x


_ArrayOrShape = TypeVar(
    "_ArrayOrShape",
    np.ndarray,
    jnp.ndarray,
    list[int],
    tuple[int, ...],
)


def size_at(
    x: _ArrayOrShape | core.ShapedArray, axes: Iterable[int] | None = None
) -> int:
    if hasattr(x, "shape"):
        x = x.shape

    if axes is None:
        axes = range(len(x))

    return functools.reduce(operator.mul, [x[a] for a in axes], 1)


def _read_keys(key, x1, x2):
    """Read dropout key.

    `key` might be a tuple of two rng keys or a single rng key or None. In
    either case, `key` will be mapped into two rng keys `key1` and `key2` to
    make sure `(x1==x2) == (key1==key2)`.
    """

    if key is None or all_none(x2):
        key1 = key2 = key
    elif isinstance(key, tuple) and len(key) == 2:
        key1, key2 = key
        new_key = jnp.where(x1_is_x2(key1, key2), random.fold_in(key2, 1), key2)
        key2 = jnp.where(x1_is_x2(x1, x2), key1, new_key)
        warnings.warn(
            "The value of `key[1]` might be replaced by a new value if "
            "key[0] == key[1] and x1 != x2 or key[0] != key[1] and "
            "x1 == x2."
        )
    elif isinstance(key, jnp.ndarray):
        key1 = key
        key2 = jnp.where(x1_is_x2(x1, x2), key1, random.fold_in(key, 1))
    else:
        raise TypeError(type(key))
    return key1, key2


def split_kwargs(kwargs, x1=None, x2=None):
    """Splitting `kwargs`.

    Specifically,
      1. if kwarg is a rng key, it will be split into two keys.
      2. else if it is a tuple of length two, the tuple will be split into two
         parts, one for kwargs1 and the other for kwargs2.
      3. else it is copied to kwargs1 and kwargs2.

    """
    kwargs1 = {}
    kwargs2 = {}
    for k, v in kwargs.items():
        if x2 is not None and k == "rng":
            key1, key2 = _read_keys(v, x1, x2)
            kwargs1[k] = key1
            kwargs2[k] = key2
        elif isinstance(v, tuple) and len(v) == 2:
            kwargs1[k] = v[0]
            kwargs2[k] = v[1]
        else:
            kwargs1[k] = kwargs2[k] = v

    return kwargs1, kwargs2
