# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, Any, List, Tuple, Dict, Optional
from functools import partial
from numbers import Number

import numpy as np

from .dispatch import dispatch
from .numbers import is_scalar
from .types import Array, DType


def raise_if_len_not_match(length, expected_length, string):
    if length != expected_length:
        raise ValueError(
            f"""
            Length mismatch: expected object of length {expected_length}, but
            got object of length {length} for key {string}.
            """
        )


class History:
    """
    A class to store a time-series of arbitrary data.

    An History object stores several time-series all sharing the same
    time axis. History behaves like a dictionary, where the various key
    index the values (y axis) of the time-series. The time-axis is accessed
    through the attribute `History.iters`.

    It's possible to label one time-serie (one key) as the main value, so that
    when converting to numpy array (for example for plotting with pyplot) that
    axis is automatically picked by default.

    If only one time-series is provided, without a key, then its name will
    be `value`.
    """

    def __init__(
        self,
        values: Any = None,
        iters: Optional[Union[list, Array]] = None,
        dtype: Optional[DType] = None,
        iter_dtype: Optional[DType] = None,
        main_value_name: Optional[str] = None,
    ):
        """
        Creates a new History object.

        Values should be an arbitrary type or container to initialize the
        History with.
        By default assumes that values correspond to the first iteration `0`.
        If `values` is a list or collection of lists, an array or range should
        be passed to values with the correct length.

        Optionally it's possible to specify the dtype of the data and of the
        time-axis.

        Args:
            values: a type/container of types containing the value at the first
                iteration, or an iterable/container of iterables
                containing the values at all iterations (in the latter, values
                must also be specified).
            iters: an optional iterable of iterations at which values correspond.
                If unspecified, assumes that values are logged at only one iteration.
            dtype: If no values or iters are passed, uses this dtype to store data
                if numerical
            iter_dtype: If no values or iters are passed, uses this dtype to store
                iteration numbers
            main_value_name: If data is a dict or object with to_dict method, this
                optional string labels an entry as being the main one.
        """
        single_value = False

        if values is None and iters is None:
            values = []
            iters = []
        elif iters is None:
            iters = 0

        if is_scalar(iters):
            iters = np.array([iters], dtype=iter_dtype)
        elif isinstance(iters, list):
            iters = np.array(iters, dtype=iter_dtype)

        n_elements = len(iters)

        if is_scalar(values):
            values = {"value": values}
            main_value_name = "value"
            single_value = True

        elif hasattr(values, "__array__"):
            values = {"value": values}
            main_value_name = "value"
            single_value = True

        elif hasattr(values, "to_compound"):
            main_value_name, values = values.to_compound()

        elif hasattr(values, "to_dict"):
            values = values.to_dict()

        elif isinstance(values, dict) or hasattr(values, "items"):
            pass

        else:
            values = {"value": [values]}
            main_value_name = "value"
            single_value = True

        value_dict = {"iters": iters}
        keys = []
        for (key, val) in values.items():
            if key == "iters":
                raise ValueError("cannot have a field called iters")

            if is_scalar(val):
                raise_if_len_not_match(1, n_elements, key)
                val = np.asarray([val], dtype=dtype)

            elif hasattr(val, "__array__"):
                val = np.asarray(val, dtype=dtype)
                if n_elements == 1 and len(val) != 1:
                    val = np.reshape(val, (1,) + val.shape)

                raise_if_len_not_match(len(val), n_elements, key)

            elif isinstance(val, list):
                pass
            else:
                val = [val]

            value_dict[key] = val
            keys.append(key)

        self._value_dict = value_dict
        self._value_name = main_value_name
        self._len = n_elements
        self._single_value = single_value
        self._keys = keys

    @property
    def iters(self) -> Array:
        return self._value_dict["iters"]

    @property
    def values(self) -> Array:
        return self._value_dict[self._value_name]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, key) -> Array:
        # if its an int corresponding to an element not inside the dict,
        # treat it as accessing a slice of a single element
        if isinstance(key, int) and key not in self:
            return self._get_slice(key)

        # support slice syntax
        if isinstance(key, slice):
            return self._get_slice(key)

        return self._value_dict[key]

    def _get_slice(self, slce: slice) -> "History":
        """
        get a slice of iterations from this history object
        """
        values_sliced = {}
        for key in self.keys():
            values_sliced[key] = self[key][slce]

        iters = self.iters[slce]

        hist = History(values_sliced, iters)
        hist._single_value = self._single_value
        hist._value_name = self._value_name

        return hist

    def __contains__(self, key: str) -> bool:
        return key in self._value_dict

    def keys(self) -> List:
        return self._keys

    def append(self, val: Any, it: Optional[Number] = None):
        """
        Append another value to this history object.

        Args:
            val: the value in the next timestep
            it: the time corresponding to this new value. If
                not defined, increment by 1.
        """
        append(self, val, it)

    def get(self) -> Tuple[Array, Array]:
        """
        Returns a tuple containing times and values of this history object
        """
        return self.iters, self.values

    def to_dict(self) -> Dict:
        """
        Converts the history object to dict.

        Used for serialization
        """
        return self._value_dict

    def __array__(self, *args, **kwargs) -> Array:
        """
        Automatically transform this object to a numpy array when calling
        asarray, by only considering the values and neglecting the times.
        """
        return np.asarray(self.values, *args, **kwargs)

    def __iter__(self):
        """
        You can iterate the values in history object.

        Returns the Iterator object.
        """
        return iter(zip(self.iters, self.values))

    def __getattr__(self, attr):
        # Allow users to access fields with . accessor patterns
        if attr in self._value_dict:
            return self._value_dict[attr]

        raise AttributeError

    def __repr__(self):
        return (
            "History("
            + f"\n   keys  = {self.keys()}, "
            + f"\n   iters = {self.iters},"
            + "\n)"
        )

    def __str__(self):
        return f"History(keys={self.keys()}, n_iters={len(self.iters)})"


@dispatch
def append(self: History, val: Any):
    return append(self, val, None)


@dispatch
def append(self: History, val: History, it: Any):  # noqa: E0102, F811
    if not set(self.keys()) == set(val.keys()):
        raise ValueError("cannot concatenate MVHistories with different keys")

    if it is not None:
        raise ValueError("When concatenating histories, cannot specify the iteration.")

    for key in self.keys():
        self._value_dict[key] = np.concatenate([self[key], val[key]])

    self._value_dict["iters"] = np.concatenate([self.iters, val.iters])

    self._len = len(self) + len(val)


@dispatch
def append(self: History, values: dict, it: Any):  # noqa: E0102, F811
    for key, val in values.items():
        _vals = self._value_dict[key]

        if isinstance(_vals, list):
            _vals.append(val)
        elif isinstance(_vals, np.ndarray):
            new_shape = (len(_vals) + 1,) + _vals.shape[1:]
            # try to resize in place the buffer so that we don't reallocate
            # and if we fail, resize tby reallocating to a new buffer.
            try:
                _vals.resize(new_shape)
            except ValueError:
                _vals = np.resize(_vals, new_shape)
                self._value_dict[key] = _vals

            _vals[-1] = val
        else:
            raise TypeError(f"Unknown accumulator type {type(_vals)} for key {key}.")

    try:
        self.iters.resize(len(self.iters) + 1)
    except ValueError:
        self._value_dict["iters"] = np.resize(self.iters, (len(self.iters) + 1))

    self.iters[-1] = it
    self._len += 1


@dispatch
def append(self: History, val: Any, it: Any):  # noqa: E0102, F811
    if self._single_value and is_scalar(val) or hasattr(val, "__array__"):
        append(self, {"value": val}, it)
    elif hasattr(val, "to_compound"):
        append(self, val.to_compound()[1], it)
    elif hasattr(val, "to_dict"):
        append(self, val.to_dict(), it)
    else:
        append(self, {"value": val}, it)


def accum_in_tree(fun, tree_accum, tree, compound=True, **kwargs):
    """
    Maps all the leafs in the two trees, applying the function with the leafs of tree1
    as first argument and the leafs of tree2 as second argument
    Any additional argument after the first two is forwarded to the function call.

    This is usefull e.g. to sum the leafs of two trees

    Args:
        fun: the function to apply to all leafs
        tree1: the structure containing leafs. This can also be just a leaf
        tree2: the structure containing leafs. This can also be just a leaf
        *args: additional positional arguments passed to fun
        **kwargs: additional kw arguments passed to fun

    Returns:
        An equivalent tree, containing the result of the function call.
    """
    if tree is None:
        return tree_accum

    elif isinstance(tree, list):
        if tree_accum is None:
            tree_accum = [None for _ in range(len(tree))]

        return [
            accum_in_tree(fun, _accum, _tree, **kwargs)
            for _accum, _tree in zip(tree_accum, tree)
        ]
    elif isinstance(tree, tuple):
        if tree_accum is None:
            tree_accum = (None for _ in range(len(tree)))

        return tuple(
            accum_in_tree(fun, _accum, _tree, **kwargs)
            for _accum, _tree in zip(tree_accum, tree)
        )
    elif isinstance(tree, dict):
        if tree_accum is None:
            tree_accum = {}

        for key in tree.keys():
            tree_accum[key] = accum_in_tree(
                fun, tree_accum.get(key, None), tree[key], **kwargs
            )

        return tree_accum
    elif hasattr(tree, "to_compound") and compound:
        return fun(tree_accum, tree, **kwargs)
    elif hasattr(tree, "to_dict"):
        return accum_in_tree(fun, tree_accum, tree.to_dict(), **kwargs)
    else:
        return fun(tree_accum, tree, **kwargs)


def accum_histories(accum, data, *, step=0):
    if accum is None:
        return History(data, step)
    else:
        accum.append(data, it=step)
        return accum


accum_histories_in_tree = partial(accum_in_tree, accum_histories)
