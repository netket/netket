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

from typing import Union
import numpy as np

from netket.utils import struct


# Mark this class a NetKet dataclass so that it can automatically be serialized by Flax.
@struct.dataclass(_frozen=False)
class InvalidLossStopping:
    """A simple callback to stop NetKet if there are no more improvements in the training.
    based on `driver._loss_name`."""

    monitor: str = "mean"
    """Loss statistic to monitor. Should be one of 'mean', 'variance', 'sigma'."""
    patience: Union[int, float] = 0
    """Number of epochs with invalid loss after which training will be stopped."""

    _last_valid_iter: int = 0
    """Last valid iteration, to check against patience"""

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
        if not np.isfinite(loss):
            if step - self._last_valid_iter >= self.patience:
                return False
        else:
            self._last_valid_iter = step
        return True
