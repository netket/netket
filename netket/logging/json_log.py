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

import time

import os
from os import path as _path

from flax import serialization

from netket.jax.sharding import extract_replicated

from .runtime_log import RuntimeLog


class JsonLog(RuntimeLog):
    """
      This logger serializes expectation values and other log data to a JSON file and can save the latest model parameters in MessagePack encoding to a separate file.

    It can be passed with keyword argument `out` to Monte Carlo drivers in order
    to serialize the output data of the simulation.

    This logger inherits from :class:`netket.logging.RuntimeLog`, so it maintains the dictionary
    of all logged quantities in memory, which can be accessed through the attribute
    :attr:`~netket.logging.JsonLog.data`.

    If the model state is serialized, then it can be de-serialized using the msgpack protocol
    of flax. For more information on how to de-serialize the output, see
    `here <https://flax.readthedocs.io/en/latest/flax.serialization.html>`_.
    The target of the serialization is the variational state itself.

    Data is serialized to json as several nested dictionaries. You can deserialize
    by simply calling :func:`json.load(open(filename)) <json.load>`.
    Logged expectation values will be captured inside histories objects, so they will
    have a subfield `iter` with the iterations at which that quantity has been computed,
    then `Mean` and others.
    Complex numbers are logged as dictionaries :code:`{'real':list, 'imag':list}`.
    """

    def __init__(
        self,
        output_prefix: str,
        mode: str = "write",
        save_params_every: int = 50,
        write_every: int = 50,
        save_params: bool = True,
        autoflush_cost: float = 0.005,
    ):
        """
        Construct a Json Logger.

        Args:
            output_prefix: the name of the output files before the extension
            save_params_every: every how many iterations should machine parameters be
                flushed to file
            write_every: every how many iterations should data be flushed to file
            mode: Specify the behaviour in case the file already exists at this
                output_prefix. Options are
                - `[w]rite`: (default) overwrites file if it already exists;
                - `[x]` or `fail`: fails if file already exists;
            save_params: bool flag indicating whether variables of the variational state
                should be serialized at some interval. The output file is overwritten
                every time variables are saved again.
            autoflush_cost: Maximum fraction of runtime that can be dedicated to
                serializing data. Defaults to 0.005 (0.5 per cent)
        """
        super().__init__()

        # Shorthands for mode
        if mode == "w":
            mode = "write"
        elif mode == "a":
            mode = "append"
        elif mode == "x":
            mode = "fail"

        if not ((mode == "write") or (mode == "append") or (mode == "fail")):
            raise ValueError(
                "Mode not recognized: should be one of `[w]rite`, `[a]ppend` or"
                "`[x]`(fail)."
            )

        if mode == "append":
            raise ValueError("Append mode is no longer supported.")

        file_exists = _path.exists(output_prefix + ".log") or _path.exists(
            output_prefix + ".mpack"
        )

        if file_exists and mode == "fail":
            raise ValueError(
                "Output file already exists. Either delete it manually or"
                "change `output_prefix`."
            )

        dir_name = _path.dirname(output_prefix)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)

        self._prefix = output_prefix
        self._file_mode = mode

        self._write_every = write_every
        self._save_params_every = save_params_every
        self._old_step = 0
        self._steps_notflushed_write = 0
        self._steps_notflushed_pars = 0
        self._save_params = save_params
        self._files_open = [output_prefix + ".log", output_prefix + ".mpack"]

        self._autoflush_cost = autoflush_cost
        self._last_flush_time = time.time()
        self._last_flush_runtime = 0.0
        self._last_flush_pars_time = time.time()
        self._last_flush_pars_runtime = 0.0

        self._flush_log_time = 0.0
        self._flush_pars_time = 0.0

    def __call__(self, step, item, variational_state=None):
        old_step = self._old_step
        super().__call__(step, item, variational_state)

        # Check if the time from the last flush is higher than the maximum
        # allowed runtime cost of flushing
        elapsed_time = time.time() - self._last_flush_time
        # On windows, the precision of `time.time` is much lower than that on Linux,
        # so `elapsed_time` may be essentially zero.
        # We add 1e-7 to avoid the zero division error.
        flush_anyway = (
            self._last_flush_runtime / (elapsed_time + 1e-7) < self._autoflush_cost
        )

        if (
            self._steps_notflushed_write % self._write_every == 0
            or step == old_step - 1
            or flush_anyway
        ):
            self._flush_log()

        elapsed_time = time.time() - self._last_flush_pars_time
        flush_anyway = (
            self._last_flush_pars_runtime / (elapsed_time + 1e-7) < self._autoflush_cost
        )

        if (
            self._steps_notflushed_pars % self._save_params_every == 0
            or step == old_step - 1
            or flush_anyway
        ):
            self._flush_params(variational_state)

        self._old_step = step
        self._steps_notflushed_write += 1
        self._steps_notflushed_pars += 1

    def _flush_log(self):
        # Time how long flushing data takes.
        self._last_flush_time = time.time()
        self.serialize(self._prefix + ".log")
        self._last_flush_runtime = time.time() - self._last_flush_time

        self._flush_log_time += self._last_flush_runtime
        self._steps_notflushed_write = 0

    def _flush_params(self, variational_state):
        if not self._save_params:
            return
        if variational_state is None:
            return

        self._last_flush_pars_time = time.time()
        binary_data = serialization.to_bytes(
            extract_replicated(variational_state.variables)
        )
        with open(self._prefix + ".mpack", "wb") as outfile:
            outfile.write(binary_data)
        self._last_flush_pars_runtime = time.time() - self._last_flush_pars_time

        self._flush_pars_time += self._last_flush_pars_runtime
        self._steps_notflushed_pars = 0

    def flush(self, variational_state=None):
        """
        Writes to file the content of this logger.

        Args:
            variational_state: optionally also writes the parameters of the machine.
        """
        self._flush_log()

        if variational_state is not None:
            self._flush_params(variational_state)

    def __del__(self):
        if hasattr(self, "_steps_notflushed_write"):
            if self._steps_notflushed_write > 0:
                self.flush()
        if hasattr(self, "_steps_notflushed_pars"):
            if self._steps_notflushed_pars > 0:
                self.flush()

    def __repr__(self):
        _str = f"JsonLog('{self._prefix}', mode={self._file_mode}, "
        _str = _str + f"autoflush_cost={self._autoflush_cost})"
        _str = _str + "\n  Runtime cost:"
        _str = _str + f"\n  \tLog:    {self._flush_log_time}"
        _str = _str + f"\n  \tParams: {self._flush_pars_time}"
        return _str
