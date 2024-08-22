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


from netket.operator import Squared, AbstractSuperOperator
from netket.vqs import MCMixedState
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)
from netket.jax import tree_cast
from netket.utils.types import Optimizer

from .abstract_variational_driver import AbstractVariationalDriver


class SteadyState(AbstractVariationalDriver):
    """
    Steady-state driver minimizing L^†L.
    """

    def __init__(
        self,
        lindbladian: AbstractSuperOperator,
        optimizer: Optimizer,
        *,
        variational_state: MCMixedState,
        preconditioner: PreconditionerT = identity_preconditioner,
    ):
        """
        Initializes the driver class.

        Args:
            lindbladian: The Lindbladian of the system.
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient.
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no preconditioner
                is used and the bare gradient is passed to the optimizer.
        """
        if not isinstance(lindbladian, AbstractSuperOperator):
            raise TypeError("The first argument must be a super-operator")

        super().__init__(variational_state, optimizer, minimized_quantity_name="LdagL")

        self._lind = lindbladian
        self._ldag_l = Squared(lindbladian)

        self.preconditioner = preconditioner

        self._dp = None
        self._S = None
        self._sr_info = None

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self.state.reset()

        # Compute the local energy estimator and average Energy
        self._loss_stats, self._loss_grad = self.state.expect_and_grad(self._ldag_l)

        # if it's the identity it does
        # self._dp = self._loss_grad
        self._dp = self.preconditioner(self.state, self._loss_grad, self.step_count)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = tree_cast(self._dp, self.state.parameters)

        return self._dp

    @property
    def preconditioner(self):
        """
        The preconditioner used to modify the gradient.

        This is a function with the following signature

        .. code-block:: python

            precondtioner(vstate: VariationalState,
                          grad: PyTree,
                          step: Optional[Scalar] = None)

        Where the first argument is a variational state, the second argument
        is the PyTree of the gradient to precondition and the last optional
        argument is the step, used to change some parameters along the
        optimisation.

        Often, this is taken to be :func:`~netket.optimizer.SR`. If it is
        set to `None`, then the identity is used.
        """
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, val: PreconditionerT | None):
        if val is None:
            val = identity_preconditioner

        self._preconditioner = val

    @property
    def ldagl(self):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    def __repr__(self):
        return (
            "SteadyState("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )
