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


# Mark this class a NetKet dataclass so that it can automatically be serialized by Flax.
class InvalidLossStopping(struct.Pytree, mutable=True):
    """A simple callback to stop the optimisation when the monitored quantity becomes
    invalid for at least `patience` steps.
    """

    monitor: str
    """Loss statistic to monitor. Should be one of 'mean', 'variance', 'error_of_mean'."""
    patience: int | float
    """Number of epochs with invalid loss after which training will be stopped."""

    # caches
    _last_valid_iter: int
    """Last valid iteration, to check against patience"""

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

        # caches
        self._last_valid_iter = 0

    def __call__(self, step, log_data, driver):
        """
        A boolean function that determines whether or not to stop training.

        Args:
            step: An integer corresponding to the step (iteration or epoch) in training.
            log_data: A dictionary containing log data for training.
            driver: A NetKet variational driver.

        Returns:
            A boolean. If True, training continues, else, it does not.
        """
        # clears the _last_valid_iter in case the driver was reset
        if driver.step_count < self._last_valid_iter:
            self._last_valid_iter = 0

        if driver._loss_stats is not None:
            loss = np.real(getattr(driver._loss_stats, self.monitor))

            if not np.isfinite(loss):
                if driver.step_count - self._last_valid_iter >= self.patience:
                    return False
            else:
                self._last_valid_iter = driver.step_count
        return True
