# Copyright 2020, 2021 The NetKet Authors - All rights reserved.
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

from netket.utils import struct

from netket._src.callbacks.base import AbstractCallback, StopRun


class Timeout(AbstractCallback, mutable=True):
    """A simple callback to stop NetKet after some time has passed.

    This callback monitors whether a driver has been training for more
    than a given timeout in order to hard stop training.
    """

    timeout: float = struct.field(pytree_node=False)
    """Number of seconds to wait before the training will be stopped."""

    _init_time: float | None = struct.field(
        pytree_node=False, serialize=False, default=None
    )
    """Internal field storing the time at which the run started."""

    def __init__(self, timeout: float):
        """
        Stops the optimisation after a certain time interval.

        Args:
            timeout: number of seconds after which the optimisation will
                be stopped.
        """
        if not timeout > 0:
            raise ValueError("`timeout` must be larger than 0.")
        self.timeout = timeout
        self._init_time = None

    def on_run_start(self, step, driver):
        self._init_time = time.time()

    def on_step_end(self, step, log_data, driver):
        if time.time() - self._init_time >= self.timeout:
            raise StopRun(
                f"Timeout: training stopped after {self.timeout:.1f} seconds."
            )
