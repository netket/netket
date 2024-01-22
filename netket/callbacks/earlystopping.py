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

from typing import Optional, Union

import numpy as np

from netket.utils import struct


# Mark this class a NetKet dataclass so that it can automatically be serialized by Flax.
@struct.dataclass(_frozen=False)
class EarlyStopping:
    """A simple callback to stop NetKet if there are no more improvements in the training.
    based on `driver._loss_name`.
    """

    min_delta: float = 0.0
    """Minimum change in the monitored quantity to qualify as an improvement."""
    min_reldelta: float = 0.0
    """Minimum relative change in the monitored quantity to qualify as an improvement.

    This behaves similarly to `min_delta` but is more useful for intensive quantities that
    converge to 0, where absolute tolerances might not be effective.
    """
    patience: Union[int, float] = 0
    """Number of epochs with no improvement after which training will be stopped."""
    baseline: Optional[float] = None
    """Baseline value for the monitored quantity. Training will stop if the driver hits the baseline."""
    monitor: str = "mean"
    """Loss statistic to monitor. Should be one of 'mean', 'variance', 'sigma'."""

    # The quantities below are internal and should not be edited directly
    # by the user

    _best_val: float = np.inf
    """Best value of the loss observed up to this iteration. """
    _best_iter: int = 0
    """Iteration at which the `_best_val` was observed."""
    _best_patience_counter: int = 0
    """Stores the iteration at which we've seen the best loss so far"""

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
