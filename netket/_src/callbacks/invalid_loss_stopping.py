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

import numpy as np

from netket.utils import struct

from netket._src.callbacks.base import AbstractCallback, StopRun


class InvalidLossStopping(AbstractCallback, mutable=True):
    """A simple callback to stop the optimisation when the monitored quantity becomes
    invalid for at least `patience` steps.
    """

    monitor: str = struct.field(pytree_node=False)
    """Loss statistic to monitor. Should be one of 'mean', 'variance', 'error_of_mean'."""
    patience: int | float = struct.field(pytree_node=False)
    """Number of epochs with invalid loss after which training will be stopped."""

    _last_valid_iter: int = struct.field(pytree_node=False, serialize=False, default=0)
    """Last valid iteration, to check against patience."""

    def __init__(self, monitor: str = "mean", patience: int | float = 0):
        """
        Construct a callback stopping the optimisation when the monitored quantity
        becomes invalid for at least `patience` steps.

        Args:
            monitor: a string with the name of the quantity to be monitored. This
                is applied to the standard loss optimised by a driver, such as the
                Energy for the VMC driver. Should be one of
                'mean', 'variance', 'error_of_mean' (default: 'mean').
            patience: Number of steps to wait before stopping the execution after
                the tracked quantity becomes invalid (default 0, meaning that it
                stops immediately).
        """
        self.monitor = monitor
        self.patience = patience
        self._last_valid_iter = 0

    def on_step_end(self, step, log_data, driver):
        # reset if the driver was restarted
        if driver.step_count < self._last_valid_iter:
            self._last_valid_iter = 0

        if driver._loss_stats is not None:
            loss = np.real(getattr(driver._loss_stats, self.monitor))

            if not np.isfinite(loss):
                if driver.step_count - self._last_valid_iter >= self.patience:
                    raise StopRun(
                        f"InvalidLossStopping: loss is not finite ({loss}) "
                        f"for {driver.step_count - self._last_valid_iter} steps."
                    )
            else:
                self._last_valid_iter = driver.step_count
