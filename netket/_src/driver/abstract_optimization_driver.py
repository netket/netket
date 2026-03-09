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

from typing import Any
from functools import partial

import jax

from netket.utils import struct
from netket.utils.types import Optimizer
from netket.vqs import VariationalState

from netket._src.driver.abstract_variational_driver import AbstractDriver


class AbstractOptimizationDriver(AbstractDriver):
    """
    Base class for variational **optimization** drivers (VMC, SR, Infidelity, …).

    Adds an `optax <https://optax.readthedocs.io/en/latest/>`_ optimizer and
    implements :meth:`update_parameters` via :func:`apply_gradient`.

    Subclass this (not :class:`AbstractVariationalDriver` directly) when
    implementing a new optimization driver.
    """

    _optimizer: Optimizer = struct.field(pytree_node=False, serialize=False)
    _optimizer_state: Any = struct.field(pytree_node=True, serialize=True)

    def __init__(
        self,
        variational_state: VariationalState,
        optimizer: Optimizer,
        minimized_quantity_name: str = "loss",
    ):
        """
        Initializes a variational optimization driver.

        Args:
            variational_state: The variational state to be optimized.
            optimizer: an `optax <https://optax.readthedocs.io/en/latest/>`_ optimizer.
            minimized_quantity_name: the name of the loss function in
                the logged data set.
        """
        super().__init__(
            variational_state, minimized_quantity_name=minimized_quantity_name
        )
        self.optimizer = optimizer

    @property
    def optimizer(self):
        """
        The optimizer used to update the parameters at every iteration.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_state = optimizer.init(self.state.parameters)

    def update_parameters(self, dp):
        """
        Updates the parameters of the machine using the optimizer in this driver.

        Args:
            dp: the pytree containing the updates to the parameters.
        """
        self._optimizer_state, self.state.parameters = apply_gradient(
            self._optimizer.update, self._optimizer_state, dp, self.state.parameters
        )


@partial(jax.jit, static_argnums=0)
def apply_gradient(optimizer_fun, optimizer_state, dp, params):
    import optax

    updates, new_optimizer_state = optimizer_fun(dp, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_optimizer_state, new_params
