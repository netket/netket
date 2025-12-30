# Copyright 2021 The NetKet Authors - All rights reserved.
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

from netket.utils import struct
from netket.utils.types import PyTree
import jax.numpy as jnp


class BestLoss(struct.Pytree, mutable=True):
    """
    Callback that retains the best value of the loss function and the corresponding best variables.
    """

    best_loss: float
    """best_loss (float): The best (lowest) loss value encountered during training."""
    best_variables: PyTree
    """best_variables (PyTree): The model variables corresponding to the best loss value."""

    def __init__(
        self,
    ):
        """
        Constructs a callback retaining the best loss and the corresponding variables.
        """

        self.best_loss = jnp.inf
        self.best_variables = None

    def __call__(self, step, log_data, driver):
        """
        A boolean function that updates the best loss and the best variables if the current
        value of the loss is below the best value within its error.
        """
        if (
            driver._loss_stats.mean + driver._loss_stats.error_of_mean
        ) < self.best_loss:
            self.best_variables = driver.state.variables
            self.best_loss = driver._loss_stats.mean

        return True
