# Copyright 2022 The NetKet Authors - All rights reserved.
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

import os

import numpy as np
from typing import Any

import jax

from netket._src.callbacks.base import AbstractCallback
from netket.utils import struct
from netket.utils.tree_walk import walk_tree_with_path

_MODE_SHORTHANDS = {"write": "w", "append": "a", "fail": "x"}
_UNSET = object()
_DEFAULT_CHUNK_BYTES = 256 * 1024
_DEFAULT_CHUNK_ROWS_MAX = 1024


def _compute_chunk_shape(value: np.ndarray, chunk_size: int | None) -> tuple[int, ...]:
    if chunk_size is not None:
        rows = chunk_size
    else:
        row_nbytes = max(1, value.nbytes)
        rows = min(_DEFAULT_CHUNK_ROWS_MAX, _DEFAULT_CHUNK_BYTES // row_nbytes)

    rows = max(1, rows)
    return (rows, *value.shape)


def _append_dataset(root, value, data, *, chunk_size: int | None = None):
    value = np.asarray(value)
    if root in data:
        f_value = data[root]
        f_value.resize(f_value.shape[0] + 1, axis=0)
        f_value[-1] = value
    else:
        maxshape = (None, *value.shape)
        chunks = _compute_chunk_shape(value, chunk_size)
        data.create_dataset(root, data=[value], maxshape=maxshape, chunks=chunks)


def _tree_log_leaf(root, tree, data, *, iter=None, chunk_size: int | None = None):
    if iter is not None:
        _append_dataset(f"{root}/iter", iter, data, chunk_size=chunk_size)
        root = f"{root}/value"

    _append_dataset(root, tree, data, chunk_size=chunk_size)
    return root


def _enter_hdf5_group(root, tree, data, *, iter=None, chunk_size=None):
    if iter is None:
        return None

    if (
        isinstance(tree, tuple)
        or hasattr(tree, "to_compound")
        or hasattr(tree, "to_dict")
    ):
        _append_dataset(f"{root}/iter", iter, data, chunk_size=chunk_size)
        return {"iter": None}

    return None


def tree_log(tree, root, data, *, iter=None, chunk_size: int | None = None):
    """
    Maps all elements in tree, recursively calling tree_log with a new root string,
    and when it reaches leaves adds them to the `data` inplace.

    Args:
        tree: a pytree where the leaf nodes contain data
        root: the root of the tags used to log to HDF5
        data: an HDF5 file modified in place
        iter: an integer number specifying at which iteration the data was generated
    """
    walk_tree_with_path(
        tree,
        root,
        visit_leaf=_tree_log_leaf,
        enter_node=_enter_hdf5_group,
        data=data,
        iter=iter,
        chunk_size=chunk_size,
    )


class HDF5Log(AbstractCallback):
    r"""
    HDF5 Logger, that can be passed with keyword argument `logger` to Monte
    Carlo drivers in order to serialize the output data of the simulation.

    The logger has support for scalar numbers, NumPy/JAX arrays, and netket.stats.Stats objects.
    These are stored as individual groups within a HDF5 file, under the main group `data/`:

    - scalars are stored as a group with one dataset values of shape :code:`(n_steps,)` containing the logged values,
    - arrays are stored in the same way, but with values having shape :code:`(n_steps, *array_shape)`,
    - netket.stats.Stats are stored as a group containing each field :code:`(Mean, Variance, etc...)` as a separate dataset.

    Importantly, each group has a dataset :code:`iters`, which tracks the
    iteration number of the logged quantity.

    Data can be deserialized by calling :code:`f = h5py.File(filename, 'r')` and
    inspecting the datasets as a dictionary, i.e. :code:`f['data/energy/Mean']`

    This class is a full :class:`~netket.callbacks.AbstractCallback` and can be passed
    either as ``out=logger`` *or* inside the ``callbacks=[..., logger]`` list.

    .. tip::
        Use the ``metadata`` argument to attach a flat dict of hyper-parameters
        (learning rate, system size, model type, …) to the output file.  They are
        stored as HDF5 attributes on the ``metadata/`` group and travel with the file,
        making it easy to correlate results without relying on external bookkeeping.

    Examples:
        Basic usage as an output logger.

        >>> import pytest; pytest.skip("skip automated test of this docstring")
        >>>
        >>> import netket as nk
        >>> logger = nk.logging.HDF5Log("output")
        >>> gs.run(n_iter=300, out=logger)
        >>> # data lives in output.h5

        Attaching metadata to record hyper-parameters.

        >>> import pytest; pytest.skip("skip automated test of this docstring")
        >>>
        >>> import netket as nk
        >>> logger = nk.logging.HDF5Log(
        ...     "output",
        ...     metadata={"learning_rate": 0.01, "alpha": 1, "L": 20},
        ... )
        >>> gs.run(n_iter=300, out=logger)
        >>> # f['metadata'].attrs['learning_rate'] == '0.01'

        Using the logger as a callback.

        >>> import pytest; pytest.skip("skip automated test of this docstring")
        >>>
        >>> import netket as nk
        >>> logger = nk.logging.HDF5Log("output")
        >>> gs.run(n_iter=300, callbacks=[logger])

    .. note::
        The API of this logger is covered by our Semantic Versioning API guarantees. However, the structure of the
        logged files is not, and might change in the future. If you think that we could improve the output format of
        this logger, please open an issue on the NetKet repository and let us know.

    """

    _metadata: Any = struct.field(pytree_node=False, serialize=False)
    _file_mode: Any = struct.field(pytree_node=False, serialize=False)
    _file_name: Any = struct.field(pytree_node=False, serialize=False)
    _writer: Any = struct.field(pytree_node=False, serialize=False)
    _write_every: Any = struct.field(pytree_node=False, serialize=False)
    _steps_notflushed_write: Any = struct.field(pytree_node=False, serialize=False)
    _chunk_size: Any = struct.field(pytree_node=False, serialize=False)

    def __init__(
        self,
        path: str,
        mode: str = "write",
        save_params=_UNSET,
        save_params_every=_UNSET,
        write_every: int = 50,
        chunk_size: int | None = None,
        metadata: dict | None = None,
    ):
        """
        Construct a HDF5 Logger.

        Args:
            path: the name of the output files before the extension
            mode: Specify the behaviour in case the file already exists at this
                path. Options are
                - `[w]rite`: (default) overwrites file if it already exists;
                - `[a]ppend`: appends to an existing file, otherwise creates one;
                - `[x]` or `fail`: fails if file already exists;
            save_params: deprecated, has no effect.
            save_params_every: deprecated, has no effect.
            write_every: every how many iterations the HDF5 file should be flushed
                to disk
            chunk_size: number of log entries per HDF5 chunk. If omitted, chunking
                is chosen adaptively to target a moderate chunk size in bytes.
            metadata: optional flat dict of key/value pairs stored once at run
                start as HDF5 attributes on the ``metadata/`` group.
        """
        import warnings
        import h5py  # noqa: F401

        super().__init__()
        if save_params is not _UNSET or save_params_every is not _UNSET:
            warnings.warn(
                "save_params and save_params_every are deprecated and have no effect. "
                "HDF5Log no longer serializes variational state parameters.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._metadata = metadata or {}

        if mode == "w":
            mode = "write"
        elif mode == "a":
            mode = "append"
        elif mode == "x":
            mode = "fail"

        if mode not in _MODE_SHORTHANDS:
            raise ValueError(
                "Mode not recognized: should be one of `[w]rite`, `[a]ppend` or"
                "`[x]`(fail)."
            )
        if write_every <= 0:
            raise ValueError("write_every must be a positive integer.")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer or None.")

        mode = _MODE_SHORTHANDS[mode]

        if not path.endswith((".h5", ".hdf5")):
            path = path + ".h5"

        if os.path.exists(path) and mode == "x":
            raise ValueError(
                "Output file already exists. Either delete it manually or"
                "change `path`."
            )

        dir_name = os.path.dirname(path)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)

        self._file_mode = mode
        self._file_name = path
        self._writer = None

        self._write_every = write_every
        self._steps_notflushed_write = 0
        self._chunk_size = chunk_size

    def _init_output_file(self):
        if self._is_master_process:
            import h5py

            self._writer = h5py.File(self._file_name, self._file_mode)
            self._file_mode = "a"  # subsequent opens always append

    def __call__(self, step, log_data, variational_state=None):
        if self._writer is None:
            self._init_output_file()

        if self._is_master_process:
            tree_log(
                log_data,
                "data",
                self._writer,
                iter=step,
                chunk_size=self._chunk_size,
            )

        self._steps_notflushed_write += 1
        if self._steps_notflushed_write >= self._write_every:
            self.flush()

    def flush(self):
        """Writes buffered data to disk."""
        if self._writer is not None:
            self._writer.flush()
        self._steps_notflushed_write = 0

    def _close(self):
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __del__(self):
        try:
            self._close()
        except Exception:
            pass

    def __repr__(self):
        _str = f"HDF5Log('{self._file_name}', mode={self._file_mode}"
        _str += f", write_every={self._write_every}, chunk_size={self._chunk_size})"
        return _str

    # --- AbstractLog compatibility ---

    @property
    def _is_master_process(self) -> bool:
        return jax.process_index() == 0

    # --- AbstractCallback interface ---

    @property
    def callback_order(self) -> int:
        return 10

    def on_run_start(self, step, driver):
        if self._writer is None:
            self._init_output_file()
        if self._metadata and self._is_master_process:
            grp = self._writer.require_group("metadata")
            for key, val in self._metadata.items():
                grp.attrs[key] = val

    def on_step_end(self, step, log_data, driver):
        self(step, log_data)

    def on_run_end(self, step, driver):
        self._close()

    def on_run_error(self, step, error, driver):
        self._close()
