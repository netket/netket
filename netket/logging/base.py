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

import abc

from typing import Any

import jax

from netket.vqs import VariationalState
from netket.utils import mpi


class AbstractLog(abc.ABC):
    """
    Abstract base class detailing the interface that loggers must
    implement in order to work with netket drivers.

    It can be passed with keyword argument `out` to Monte Carlo drivers in order
    to serialize the output data of the simulation.

    .. note::

        When using Loggers in a multi-process (MPI or Sharding) context, some care
        must be taken to ensure that they work correctly.

        The design philosophy adopted by NetKet follows the Jax/Orbax convention
        that the same code should be executd on all processes. Therefore, loggers
        should expect to be executed and called from ALL processes, and it is
        their responsability to only perform expensive I/O operations on the root
        rank.

        They can verify if they are running on the root rank by calling
        `self._is_master_process`.

        Have a look at :class:`netket.logging.RuntimeLog` or
        :class:`netket.logging.TensorBoardLog` for a good example.

    """

    @abc.abstractmethod
    def __call__(
        self,
        step: int,
        item: dict[str, Any],
        variational_state: VariationalState | None = None,
    ):
        """
        Logs at a given integer step a dictionary of data, optionally
        specifying a variational state to encode additional data.

        Args:
            step: monotonically increasing integer representing the row in the
                database corresponding to this log entry;
            item: Any dictionary of data to be logged;
            variational_state: optional variational state from which additional data
                might be extracted.
        """

    @abc.abstractmethod
    def flush(self, variational_state: VariationalState | None = None):
        """
        Flushes the data that is stored internally to disk/network.

        Args:
            variational_state: optional variational state from which additional data
                might be extracted.

        """

    @property
    def _is_master_process(self) -> bool:
        """
        Returns whether this logger is the root logger in a distributed setting.
        """
        return mpi.rank == 0 and jax.process_index() == 0
