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

from typing import Any, IO, TYPE_CHECKING
from pathlib import Path

import jax
import numpy as np
import orjson

from netket.utils.history import accum_histories_in_tree, HistoryDict

if TYPE_CHECKING:
    from netket.vqs import VariationalState

from .base import AbstractLog


class RuntimeLog(AbstractLog):
    """
    This logger accumulates log data in a set of nested dictionaries which are stored in memory. The log data is not automatically saved to the filesystem.

    It can be passed with keyword argument `out` to Monte Carlo drivers in order
    to serialize the output data of the simulation.

    This logger keeps the data in memory, and does not save it to disk. To serialize
    the current content to a file, use the method :py:meth:`~netket.logging.RuntimeLog.serialize`.
    """

    _data: dict[str, Any]

    def __init__(self):
        """
        Crates a Runtime Logger.
        """
        self._data: dict[str, Any] = HistoryDict()
        self._old_step = 0

    def __call__(
        self,
        step: int,
        item: dict[str, Any],
        variational_state: "VariationalState | None" = None,
    ):
        if self._data is None:
            self._data = {}
        self._data = accum_histories_in_tree(self._data, item, step=step)
        self._old_step = step

    @property
    def data(self) -> dict[str, Any]:
        """
        The dictionary of logged data.
        """
        return self._data

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def flush(self, variational_state=None):
        pass

    @classmethod
    def deserialize(cls, path: str | Path) -> "RuntimeLog":
        r"""
        Load a :class:`~netket.logging.RuntimeLog` from a file previously serialized with
        :py:meth:`~netket.logging.RuntimeLog.serialize`.

        If the path has no extension, ``.json`` is appended, consistent with the
        behaviour of :py:meth:`~netket.logging.RuntimeLog.serialize`.

        Args:
            path: The path of the file to load.

        Returns:
            A new :class:`~netket.logging.RuntimeLog` instance with the data from the file.
        """
        if isinstance(path, str):
            path = Path(path)

        if isinstance(path, Path):
            filename = path.name
            if not filename.endswith((".log", ".json")):
                path = path.parent / (filename + ".json")

        data = HistoryDict.from_file(path)

        log = cls.__new__(cls)
        log._data = data

        # Recover _old_step from the maximum last iter across all histories
        def _find_last_step(d):
            last = 0
            for val in d.values():
                if isinstance(val, dict):
                    last = max(last, _find_last_step(val))
                elif hasattr(val, "iters") and len(val.iters) > 0:
                    last = max(last, int(val.iters[-1]))
            return last

        log._old_step = _find_last_step(data._data)

        return log

    def serialize(self, path: str | Path | IO):
        r"""
        Serialize the content of :py:attr:`~netket.logging.RuntimeLog.data` to a file.

        If the file already exists, it is overwritten.

        Args:
            path: The path of the output file. It must be a valid path.
        """
        if isinstance(path, str):
            path = Path(path)

        if not self._is_master_process:
            return

        if isinstance(path, Path):
            parent = path.parent
            filename = path.name
            if not filename.endswith((".log", ".json")):
                filename = filename + ".json"
            path = parent / filename

            with open(path, "wb") as io:
                self._serialize(io)
        else:
            self._serialize(path)

    def _serialize(self, outstream: IO):
        r"""
        Inner method of `serialize`, working on an IO object.
        """
        outstream.write(
            orjson.dumps(
                self.data,
                default=default,
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
        )

    def __repr__(self):
        _str = "RuntimeLog():\n"
        if self.data is not None:
            _str += f" keys = {list(self.data.keys())}"
        return _str


def default(obj):
    if hasattr(obj, "to_json"):
        return obj.to_json()
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.complexfloating):
            return {
                "real": np.ascontiguousarray(obj.real),
                "imag": np.ascontiguousarray(obj.imag),
            }
        else:
            return np.ascontiguousarray(obj)
    elif isinstance(obj, jax.numpy.ndarray):
        return np.ascontiguousarray(obj)
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}

    raise TypeError
