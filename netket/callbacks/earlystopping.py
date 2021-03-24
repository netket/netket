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

from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class EarlyStopping:
    """A simple callback to stop NetKet if there are no more improvements in the training.
    based on `driver._loss_name`."""

    min_delta: float = 0.0
    """Minimum change in the monitored quantity to qualify as an improvement."""
    patience: Union[int, float] = 0
    """Number of epochs with no improvement after which training will be stopped."""
    baseline: float = None
    """Baseline value for the monitored quantity. Training will stop if the driver hits the baseline."""
    monitor: str = "mean"
    """Loss statistic to monitor. Should be one of 'mean', 'variance', 'sigma'."""

    def __post_init__(self):
        self._best_val = np.infty
        self._best_iter = 0

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
        if loss <= self._best_val:
            self._best_val = loss
            self._best_iter = step
        if self.baseline is not None:
            if loss <= self.baseline:
                return False
        if (
            step - self._best_iter >= self.patience
            and loss > self._best_val - self.min_delta
        ):
            return False
        else:
            return True
