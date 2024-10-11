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


class EarlyStopping(struct.Pytree, mutable=True):
    """A simple callback to stop NetKet if there are no more improvements in the training.
    based on `driver._loss_name`.
    """

    min_delta: float
    """Minimum change in the monitored quantity to qualify as an improvement."""
    min_reldelta: float
    """Minimum relative change in the monitored quantity to qualify as an improvement.

    This behaves similarly to `min_delta` but is more useful for intensive quantities that
    converge to 0, where absolute tolerances might not be effective.
    """
    patience: int | float
    """Number of epochs with no improvement after which training will be stopped."""
    baseline: float | None
    """Baseline value for the monitored quantity. Training will stop if the driver is above the baseline."""
    monitor: str
    """Loss statistic to monitor. Should be one of 'mean', 'variance', 'error_of_mean'."""
    start_from_step: int
    """Number of steps to wait before the callback has any effect."""

    # The quantities below are internal and should not be edited directly
    # by the user

    _best_val: float = np.inf
    """Best value of the loss observed up to this iteration. """
    _best_iter: int
    """Iteration at which the `_best_val` was observed."""
    _best_patience_counter: int
    """Stores the iteration at which we've seen the best loss so far"""

    def __init__(
        self,
        min_delta: float = 0.0,
        min_reldelta: float = 0.0,
        patience: int | float = 0,
        baseline: float | None = None,
        start_from_step: int = 0,
        monitor: str = "mean",
    ):
        """
        Construct an early stopping callback.

        Args:
            min_delta: Minimum change in the monitored quantity to
                qualify as an improvement.
            min_reldelta: Minimum relative change in the monitored
                quantity to qualify as an improvement.
            patience: Number of epochs with no improvement after which
                training will be stopped.
            baseline: Baseline value for the monitored quantity. Training
                will stop if the driver does not drop below the baseline.
            monitor: Loss statistic to monitor. Should be one of
                'mean', 'variance', 'error_of_mean'.
            start_from_step: Number of steps to wait before the callback has
                any effect. Defaults to `0`.
        """
        self.min_delta = min_delta
        self.min_reldelta = min_reldelta
        self.patience = patience
        self.baseline = baseline
        self.monitor = monitor
        self.start_from_step = start_from_step

        self._best_val = np.inf
        self._best_iter = 0
        self._best_patience_counter = 0

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

        if step < self.start_from_step:
            return True

        loss = np.real(getattr(log_data[driver._loss_name], self.monitor))

        self._best_patience_counter += 1
        if self._is_improvement(loss, self._best_val):
            self._best_val = loss
            self._best_iter = step

            if self.baseline is None:
                self._best_patience_counter = 0
            elif self._is_improvement(loss, self.baseline):
                # If using baseline, update patience only if we are better than baseline
                self._best_patience_counter = 0

        if self._best_patience_counter > self.patience:
            return False

        return True

    def _is_improvement(self, loss, target):
        # minimal value for absolute and relative improvement
        abs_minval = target - self.min_delta
        rel_minval = target * (1 - self.min_reldelta)
        # minimval value that qualify as an improvement
        minval = min(abs_minval, rel_minval)

        return loss < minval
