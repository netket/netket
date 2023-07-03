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

from typing import Optional
import time

from netket.utils import struct


# Mark this class a NetKet dataclass so that it can automatically be serialized by Flax.
@struct.dataclass(_frozen=False)
class Timeout:
    """A simple callback to stop NetKet after some time has passed.

    This callback monitors whether a driver has been training for more
    than a given timeout in order to hard stop training.
    """

    timeout: float
    """Number of seconds to wait before the training will be stopped."""

    _init_time: Optional[float] = None
    """
    Internal field storing the time at which the first iteration has been
    performed.
    """

    def __post_init__(self):
        if not self.timeout > 0:
            raise ValueError("`timeout` must be larger than 0.")

    def reset(self):
        """Resets the initial time of the training"""
        self.__init_time = None

    def __call__(self, step, log_data, driver):
        """
        A boolean function that determines whether or not to stop training.

        Args:
            step: An integer corresponding to the step (iteration or epoch) in training.
            log_data: A dictionary containing log data for training.
            driver: A NetKet variational driver.

        Returns:
            A boolean. If True, training continues, else, it does not.

        Note:
            This callback does not make use of `step`, `log_data` nor `driver`.
        """
        if self._init_time is None:
            self._init_time = time.time()

        if time.time() - self._init_time >= self.timeout:
            return False
        else:
            return True
