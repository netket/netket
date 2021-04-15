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

import numpy as np
from numbers import Number
import jax.numpy as jnp

from .numbers import dtype, is_scalar


def raise_if_len_not_match(length, expected_length, string):
    if length != expected_length:
        raise ValueError(
            """
            Length mismatch: expected object of length {expected_length}, but
            got object of length {length} for key {string}.
            """
        )


class MVHistory:
    """
    A class to store a time-series of scalar data.

    It has two member variables, `iter` and `values`.
    The first stores the `time` of the time series, while `values`
    stores the values at each iteration.
    """

    def __init__(self, values=[], iters=None, dtype=None, iter_dtype=None):
        value_name = None
        single_value = False

        if iters is None:
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

        elif isinstance(values, dict) or hasattr(values, "items"):
            pass

        else:
            values = {"value": [values]}
            main_value_name = "value"
            single_value = True

        value_dict = {"iters": iters}
        for (key, val) in values.items():
            if key == "iters":
                raise ValueError("cannot have a field called iters")

            if is_scalar(val):
                raise_if_len_not_match(1, n_elements, key)
                val = np.asarray(val, dtype=dtype)

            elif hasattr(val, "__array__"):
                val = np.asarray(val, dtype=dtype)
                if n_elements == 1 and len(val) != 1:
                    val = np.reshape(val, (1,) + val.shape)

                raise_if_len_not_match(len(val), n_elements, key)

            value_dict[key] = val

        self._value_dict = value_dict
        self._value_name = main_value_name
        self._len = n_elements
        self._single_value = single_value
        self._keys = list(value_dict.keys())

    @property
    def iters(self):
        return self._value_dict["iters"]

    @property
    def values(self):
        return self._value_dict[self._value_name]

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        return self._value_dict[key]

    def __contains__(self, key):
        return key in self._value_dict

    def keys(self):
        return self._keys

    def append(self, val, it=None):
        """
        Append another value to this history object.

        Args:
            val: the value in the next timestep
            it: the time corresponding to this new value. If
                not defined, increment by 1.
        """
        if isinstance(val, MVHistory):
            if not set(self.keys()) == set(val.keys()):
                raise ValueError("cannot concatenate MVHistories with different keys")

            self._value_dict["iters"] = np.concatenate([self.iters, val.iters])

            for key in self.keys():
                self._value_dict[key] = np.concatenate([self[key], val[key]])

            self._len = len(self) + len(val)
            return

        if self._single_value and _is_scalar(val):
            val_dict = {"value": val}
        else:
            _, val_dict = val.to_compound()

        for key in val_dict.keys():
            try:
                self[key].resize(len(self) + 1)
            except:
                self._value_dict[key] = np.resize(self[key], (len(self) + 1))

            self[key][-1] = val_dict[key]

        try:
            self.iters.resize(len(self.iters) + 1)
        except:
            self._value_dict["iters"] = np.resize(self.iters, (len(self.iters) + 1))

        if it is None:
            it = self.iters[-1] - self.iters[-2]

        self.iters[-1] = it
        self._len += 1

    def get(self):
        """
        Returns a tuple containing times and values of this history object
        """
        return self.iters, self.values

    def to_dict(self):
        """
        Converts the history object to dict.

        Used for serialization
        """
        return self._value_dict

    def __array__(self, *args, **kwargs):
        """
        Automatically transform this object to a numpy array when calling
        asarray, by only considering the values and neglecting the times.
        """
        return np.array(self.values, *args, **kwargs)

    def __iter__(self):
        """
        You can iterate the values in history object.
        """
        """ Returns the Iterator object """
        return iter(zip(self.iters, self.values))

    def __getattr__(self, attr):
        # Allow users to access fields with . accessor patterns
        if attr in self._value_dict:
            return self._value_dict[attr]

        raise AttributeError

    def __repr__(self):
        return (
            "MVHistory("
            + f"\n   keys  = {self.keys()}, "
            + f"\n   iters = {self.iters},"
            + f"\n)"
        )

    def __str__(self):
        return f"MVHistory(keys={self.keys()}, n_iters={len(self.iters)})"


from functools import partial
from jax.tree_util import tree_map


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
        return History([data], step)
    else:
        accum.append(data, it=step)
        return accum


def accum_mvhistories(accum, data, *, step=0):
    if accum is None:
        return MVHistory(data, step)
    else:
        accum.append(data, it=step)
        return accum


accum_histories_in_tree = partial(accum_in_tree, accum_mvhistories)
