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

from typing import Any, TYPE_CHECKING
from collections.abc import Callable

import sys

import orjson

import numpy as np

import jax

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from .history import History
else:
    History = Any


def _format_key(k):
    if hasattr(k, "key"):
        return str(k.key)
    elif hasattr(k, "idx"):
        return str(k.idx)
    elif hasattr(k, "name"):
        return str(k, k.name)
    else:
        return str(k)  # fallback


class HistoryDict:
    """A class that behaves like a dictionary, mapping strings to History instances or nested
    HistoryDict instances.

    This class behaves to all effects as a dictionary at runtime, but it allows us to specify
    custom serialization and deserialization routines, which are used to save and load the data.

    .. note::

        This class can be used to deserialize a dictionary of histories, such as the standard
        log files obtained from :class:`~netket.logging.RuntimeLog` and :class:`~netket.logging.JsonLog`.


    .. warn::

        As of november 2024, this class is to be considered an experimental implementation detail,
        used to play well with checkpointing. Do not use directly unless you know what you are doing.

    """

    def __init__(self, *args, **kwargs):
        """
        Create a new HistoryDict instance from a dictionary of histories.
        """
        data = dict(*args, **kwargs)
        # Do not nest HistoryDict inside Historydict!
        for k, d in data.items():
            if isinstance(d, HistoryDict):
                data[k] = d.to_dict()
        self._data = data

    def __setitem__(self, key: str, value: History | Self):
        if isinstance(value, HistoryDict):
            value = value.to_dict()
        self._data[key] = value

    def __getitem__(self, key: str, *, wrap_dicts=True) -> History | Self:
        if key not in self._data and "/" in key:
            keys = key.split("/")
            val = self._data
            for subkey in keys:
                val = val[subkey]  # Traverse through subdictionaries
            if wrap_dicts and isinstance(val, dict):
                return HistoryDict(val)
            return val
        val = self._data[key]
        if wrap_dicts and isinstance(val, dict):
            return HistoryDict(val)
        return val

    def __repr__(self):
        dk, _ = jax.tree_util.tree_flatten_with_path(self._data)
        if len(dk) == 0:
            return "HistoryDict({})"

        _repr_str = f"HistoryDict with {len(dk)} elements:"
        for keys, val in dk:
            path = "/".join([_format_key(k) for k in keys])
            _repr_str += f"\n\t'{path}' -> " + str(val)

        return _repr_str

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def get(self, key: str, default=None):
        """
        Get a value from the dictionary, with a default value if the key is not present.

        If the value is a dictionary, it is wrapped in a HistoryDict instance.

        Args:
            key: The key to retrieve.
            default: The default value to return if the key is not present.
        """
        res = self._data.get(key, default)
        if isinstance(res, dict):
            return HistoryDict(res)
        return res

    def to_dict(self) -> dict[str, dict | np.ndarray]:
        """
        Convert the HistoryDict to normal dictionary, with all nested HistoryDict instances.
        """
        return {k: self.__getitem__(k, wrap_dicts=False) for k in self._data.keys()}

    @classmethod
    def from_file(cls, fname: str) -> Self:
        """
        Create an HistoryDict from a text-file containing its serialization.

        Args:
            fname: The name of the file to read.
        """
        with open(fname) as f:
            data = orjson.loads(f.read())

        data = histdict_to_nparray(data)

        def _recompose(hist_dict):
            # hist_dict = {k: np.array(v) for k, v in hist_dict.items()}
            for _, checker, reconstructor in DESERIALIZATION_REGISTRY:
                if checker(hist_dict):
                    return reconstructor(hist_dict)
            raise ValueError("No matching type found for the given dictionary.")

        def _is_leaf(dic):
            if isinstance(dic, dict):
                if "iters" in dic:
                    return True
            return False

        return cls(jax.tree.map(_recompose, data, is_leaf=_is_leaf))

    def _ipython_key_completions_(self):
        return self._data.keys()


# Loading
# A registry to map types to their precedence, checker, and reconstructor functions
DESERIALIZATION_REGISTRY = []


def register_historydict_deserialization_fun(
    checker: Callable,
    reconstructor: Callable,
    precedence: int = 0,
):
    """
    Register a type with a precedence, checker, and reconstructor.

    Args:
        checker: A callable that takes a dictionary and returns True if it matches the type.
        reconstructor: A callable that takes a dictionary and reconstructs the object.
        precedence: An integer indicating the precedence of this type (higher is checked first).
    """
    DESERIALIZATION_REGISTRY.append((precedence, checker, reconstructor))
    # Sort the registry by precedence in descending order
    DESERIALIZATION_REGISTRY.sort(key=lambda x: x[0], reverse=True)


# Checker and reconstructor for History
def is_history(hist_dict):
    return "iters" in hist_dict


def reconstruct_history(hist_dict):
    from netket.utils.history.history import History

    if "Mean" in hist_dict:
        return History(hist_dict, main_value_name="Mean")
    return History(hist_dict)


# Register History with precedence 10
register_historydict_deserialization_fun(
    is_history, reconstruct_history, precedence=-10
)


"""
Json serializer for complex numbers.

Json does not support complex numbers, which are stored as {"real": ..., "imag": ...} dictionaries.
This is a workaround to store complex numbers in json files.
"""


def _is_number_list(x):
    return (
        isinstance(x, list) and len(x) > 0 and isinstance(x[0], (int, float, complex))
    )


def _is_complex_leaf(subtree):
    # Print only if you want to debug
    # print("checking if subtree is leaf", subtree)
    return (
        isinstance(subtree, dict) and "real" in subtree and "imag" in subtree
    ) or _is_number_list(subtree)


def _convert_complex(subtree):
    if isinstance(subtree, dict) and "real" in subtree and "imag" in subtree:
        return np.array(subtree["real"]) + 1j * np.array(subtree["imag"])
    return np.array(subtree)


def histdict_to_nparray(hist_dict):
    """
    Convert the {'real': ..., 'imag': ...} dictionaries to complex numbers.
    """
    return jax.tree.map(_convert_complex, hist_dict, is_leaf=_is_complex_leaf)
