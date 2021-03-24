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

from numpy import infty
import time


class Timeout:
    """A simple callback to stop NetKet after some time has passed."""

    def __init__(self, timeout):
        """
        Constructs a new Timeout object that monitors whether a driver has been training
        for more than a given timeout in order to hard stop training.

        Args:
            timeout: Number of seconds to wait before hard stopping training.
        """
        assert timeout > 0
        self.__timeout = timeout
        self.__init_time = None

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
        if self.__init_time is None:
            self.__init_time = time.time()
        else:
            print(time.time() - self.__init_time)
            if time.time() - self.__init_time >= self.__timeout:
                return False
        return True
