# Copyright 2020, 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numpy import infty


class EarlyStopping:
    """A simple callback to stop NetKet if there are no improvements in the training"""

    def __init__(self, min_delta=0, patience=0, baseline=None):
        """
        Constructs a new EarlyStopping object that monitors whether a driver is improving
        over optimisation epochs based on `driver._loss_name`.

        Args:
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            patience: Number of epochs with no improvement after which training will be stopped.
            baseline: Baseline value for the monitored quantity. Training will stop if the driver
                hits the baseline.
        """
        self.__min_delta = min_delta
        self.__patience = patience
        self.__baseline = baseline
        self._best_val = infty
        self.__best_iter = 0

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
        loss = log_data[driver._loss_name].mean.real
        if loss <= self._best_val:
            self._best_val = loss
            self.__best_iter = step
        if self.__baseline is not None:
            if loss <= self.__baseline:
                return False
        if (
            step - self.__best_iter >= self.__patience
            and loss > self._best_val - self.__min_delta
        ):
            return False
        else:
            return True
