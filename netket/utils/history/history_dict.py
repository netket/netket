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
            path = "/".join([val.key for val in keys])
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

        from .history import History

        def _recompose(hist_dict):
            # hist_dict = {k: np.array(v) for k, v in hist_dict.items()}
            if "Mean" in hist_dict:
                return History(hist_dict, main_value_name="Mean")
            else:
                return History(hist_dict)

        def _is_leaf(dic):
            if isinstance(dic, dict):
                if "iters" in dic:
                    return True
            return False

        return cls(jax.tree.map(_recompose, data, is_leaf=_is_leaf))

    def _ipython_key_completions_(self):
        return self._data.keys()
