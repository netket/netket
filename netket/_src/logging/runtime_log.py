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

from netket.utils.history import accum_histories_in_tree, History, HistoryDict

if TYPE_CHECKING:
    from netket.vqs import VariationalState

from netket._src.logging.base import AbstractLog


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

    def on_checkpoint_save(self, step, driver):
        """Writes the logged data to disk before a checkpoint is taken."""
        self.flush(driver.state)

    def on_run_start(self, step, driver):
        """
        Drops the data logged at iteration ``step`` or later when a run starts at
        ``step``, so that a run resumed from a checkpoint does not log the steps
        after it twice, and a fresh run does not mix with the rows of an earlier one.
        """
        self._data = _drop_from(self._data, step)
        self._old_step = _last_step(self._data)

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
        log._old_step = _last_step(data)

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


def _drop_from(tree, step):
    """``tree`` without the entries of its histories logged at ``step`` or later."""
    if isinstance(tree, History):
        return tree._get_slice(np.asarray(tree.iters) < step)
    if isinstance(tree, HistoryDict):
        return HistoryDict(_drop_from(tree._data, step))
    if isinstance(tree, dict):
        return {k: _drop_from(v, step) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_drop_from(v, step) for v in tree)
    return tree


def _last_step(tree) -> int:
    """The last iteration logged in the histories of ``tree`` (0 if none)."""
    if isinstance(tree, History):
        return int(tree.iters[-1]) if len(tree.iters) > 0 else 0
    if isinstance(tree, HistoryDict):
        tree = tree._data
    if isinstance(tree, dict):
        tree = list(tree.values())
    if isinstance(tree, (list, tuple)):
        return max((_last_step(v) for v in tree), default=0)
    return 0


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
