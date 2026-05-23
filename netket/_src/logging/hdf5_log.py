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
from flax.core import FrozenDict, pop as fpop
from flax.serialization import to_bytes

from netket.jax.sharding import extract_replicated
from netket.logging.base import AbstractLog

_MODE_SHORTHANDS = {"write": "w", "append": "a", "fail": "x"}
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

    if tree is None:
        return

    elif isinstance(tree, list):
        for i, val in enumerate(tree):
            tree_log(val, f"{root}/{i}", data, iter=iter, chunk_size=chunk_size)

    # handle namedtuples
    elif isinstance(tree, tuple) and hasattr(tree, "_fields"):
        tree_log(iter, f"{root}/iter", data, chunk_size=chunk_size)
        for key in tree._fields:
            tree_log(getattr(tree, key), f"{root}/{key}", data, chunk_size=chunk_size)

    elif isinstance(tree, tuple):
        tree_log(iter, f"{root}/iter", data, chunk_size=chunk_size)
        for i, val in enumerate(tree):
            tree_log(val, f"{root}/{i}", data, chunk_size=chunk_size)

    elif isinstance(tree, dict):
        for key, value in tree.items():
            tree_log(
                value,
                f"{root}/{key}",
                data,
                iter=iter,
                chunk_size=chunk_size,
            )  # noqa: F722

    elif hasattr(tree, "to_compound"):
        tree_log(iter, f"{root}/iter", data, chunk_size=chunk_size)
        tree_log(tree.to_compound()[1], root, data, chunk_size=chunk_size)  # noqa: F722

    elif hasattr(tree, "to_dict"):
        tree_log(iter, f"{root}/iter", data, chunk_size=chunk_size)
        tree_log(tree.to_dict(), root, data, chunk_size=chunk_size)  # noqa: F722

    else:
        if iter is not None:
            tree_log(iter, f"{root}/iter", data, chunk_size=chunk_size)
            root = f"{root}/value"
        value = np.asarray(tree)
        if root in data:
            f_value = data[root]
            f_value.resize(f_value.shape[0] + 1, axis=0)
            f_value[-1] = value
        else:
            maxshape = (None, *value.shape)
            chunks = _compute_chunk_shape(value, chunk_size)
            data.create_dataset(root, data=[value], maxshape=maxshape, chunks=chunks)


class HDF5Log(AbstractLog):
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

    If the model state is serialized, then it is serialized as a dataset in the group `variational_state/`.
    The target of the serialization is the parameters PyTree of the variational state (stored in the group
    `variational_state/parameters`), and the rest of the variational state variables (stored in the group
    `variational_state/model_state`)

    Data can be deserialized by calling :code:`f = h5py.File(filename, 'r')` and
    inspecting the datasets as a dictionary, i.e. :code:`f['data/energy/Mean']`

    .. note::
        The API of this logger is covered by our Semantic Versioning API guarantees. However, the structure of the
        logged files is not, and might change in the future. If you think that we could improve the output format of
        this logger, please open an issue on the NetKet repository and let us know.

    """

    def __init__(
        self,
        path: str,
        mode: str = "write",
        save_params: bool = True,
        save_params_every: int = 1,
        write_every: int = 50,
        chunk_size: int | None = None,
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
            save_params: bool flag indicating whether variables of the variational state
                should be serialized at some interval
            save_params_every: every how many iterations should machine parameters be
                flushed to file
            write_every: every how many iterations the HDF5 file should be flushed
                to disk
            chunk_size: number of log entries per HDF5 chunk. If omitted, chunking
                is chosen adaptively to target a moderate chunk size in bytes.
        """
        import h5py  # noqa: F401

        super().__init__()

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
        if save_params_every <= 0:
            raise ValueError("save_params_every must be a positive integer.")
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
        self._closed = False

        self._save_params = save_params
        self._save_params_every = save_params_every
        self._steps_notsaved_params = 0
        self._write_every = write_every
        self._steps_notflushed_write = 0
        self._chunk_size = chunk_size
        self._last_step = None

    def _init_output_file(self):
        if self._closed:
            raise RuntimeError("Cannot use HDF5Log after it has been closed.")
        if self._is_master_process:
            import h5py

            self._writer = h5py.File(self._file_name, self._file_mode)

    def _flush_writer(self):
        if self._writer is not None:
            self._writer.flush()
        self._steps_notflushed_write = 0

    def _extract_variables(self, variational_state):
        variables = variational_state.variables

        # TODO: remove - FrozenDict are deprecated
        if isinstance(variables, FrozenDict):
            variables = variables.unfreeze()
        return extract_replicated(variables)

    def _write_variational_state(self, variational_state, *, step: int):
        if not self._save_params or variational_state is None:
            return
        if self._writer is None:
            self._init_output_file()
        if not self._is_master_process:
            self._steps_notsaved_params = 0
            return

        variables = self._extract_variables(variational_state)
        _, params = fpop(variables, "params")
        binary_data = to_bytes(variables)
        tree = {"model_state": binary_data, "parameters": params, "iter": step}
        tree_log(
            tree,
            "variational_state",
            self._writer,
            chunk_size=self._chunk_size,
        )
        self._steps_notsaved_params = 0

    def __call__(self, step, log_data, variational_state):
        if self._closed:
            raise RuntimeError("Cannot use HDF5Log after it has been closed.")
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

        should_save_params = (
            self._save_params
            and variational_state is not None
            and self._steps_notsaved_params % self._save_params_every == 0
        )
        if should_save_params:
            self._write_variational_state(variational_state, step=step)

        self._last_step = step
        self._steps_notflushed_write += 1
        if self._steps_notflushed_write >= self._write_every:
            self._flush_writer()
        self._steps_notsaved_params += 1

    def flush(self, variational_state=None):
        """
        Writes to file the content of this logger.

        Args:
            variational_state: optionally also writes the parameters of the machine.
        """
        if self._closed:
            return

        if self._save_params and variational_state is not None:
            step = 0 if self._last_step is None else self._last_step + 1
            self._write_variational_state(variational_state, step=step)

        self._flush_writer()

    def close(self, variational_state=None):
        self.flush(variational_state)
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.close()
            except Exception:
                pass

    def __repr__(self):
        _str = f"HDF5Log('{self._file_name}', mode={self._file_mode}"
        _str += f", write_every={self._write_every}, chunk_size={self._chunk_size})"
        return _str
