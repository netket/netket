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


class History:
    """
    A class to store a time-series of scalar data.

    It has two member variables, `iter` and `values`.
    The first stores the `time` of the time series, while `values`
    stores the values at each iteration.
    """

    def __init__(self, values=[], iters=None, dtype=None, iter_dtype=None):
        if isinstance(values, Number):
            values = np.array([values], dtype=dtype)

        if iters is None:
            if iter_dtype is None:
                iter_dtype = np.int32
            iters = np.arange(len(values), dtype=iter_dtype)

        elif isinstance(iters, Number):
            iters = np.array([iters], dtype=iter_dtype)

        if len(values) != len(iters):
            raise ErrorException("Not matching lengths")

        self.iters = np.array(iters, dtype=iter_dtype)
        self.values = np.array(values, dtype=dtype)

    def append(self, val, it=None):
        """
        Append another value to this history object.

        Args:
            val: the value in the next timestep
            it: the time corresponding to this new value. If
                not defined, increment by 1.
        """
        if isinstance(val, History):
            self.values = np.concatenate([self.values, val.values])
            self.iters = np.concatenate([self.iters, val.iters])
            return

        try:
            self.values.resize(len(self.values) + 1)
        except:
            self.values = np.resize(self.values, (len(self.values) + 1))

        try:
            self.iters.resize(len(self.iters) + 1)
        except:
            self.iters = np.resize(self.iters, (len(self.iters) + 1))

        if it is None:
            if len(self.iters) > 2:
                it = self.iters[-1] - self.iters[-2]
            else:
                it = len(self.iters)  # 0, 1...

        self.values[-1] = val
        self.iters[-1] = it

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
        return {"iters": self.iters, "values": self.values}

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


class MVHistory:
    """
    A class to store a time-series of scalar data.

    It has two member variables, `iter` and `values`.
    The first stores the `time` of the time series, while `values`
    stores the values at each iteration.
    """

    def __init__(self, values=[], iters=None, dtype=None, iter_dtype=None):
        _value_dict = {}
        _value_name = None
        _len = 0
        _single_value = False
        _keys = []

        if isinstance(values, Number):
            values = np.array([values], dtype=dtype)
            _value_name = "value"
            _value_dict["value"] = values
            _single_value = True
            _keys.append("value")
            _len = 1

        elif hasattr(values, "to_compound"):
            _value_name, value_dict = values.to_compound()
            _value_dict = {}
            _len = 1
            for (key, val) in value_dict.items():
                _value_dict[key] = np.array([val], dtype=dtype)
                _keys.append(key)

        if iters is None:
            if iter_dtype is None:
                iter_dtype = np.int32
            _value_dict["iters"] = np.arange(_len, dtype=iter_dtype)

        elif isinstance(iters, Number):
            if _len != 1:
                raise ValueError("Need at least one iteration")
            _value_dict["iters"] = np.array([iters], dtype=iter_dtype)

        if _len != len(_value_dict["iters"]):
            raise ErrorException("Not matching lengths")

        self._value_dict = _value_dict
        self._value_name = _value_name
        self._len = _len
        self._single_value = _single_value
        self._keys = _keys

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

        if self._single_value and isinstance(val, Number):
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
