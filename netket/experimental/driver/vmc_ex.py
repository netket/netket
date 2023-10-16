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

from typing import Iterable

import jax
import jax.numpy as jnp


from netket.operator import AbstractOperator
from netket.stats import Stats
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)

from netket.driver import VMC
from netket.driver.vmc_common import info
from netket.vqs import VariationalState

from netket.experimental.excited import OperatorWithPenalty


class VMC_ex(VMC):
    """
    (Energy + penalty) minimization using Variational Monte Carlo (VMC).
    """

    # TODO docstring
    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer,
        *,
        variational_state: VariationalState,
        preconditioner: PreconditionerT = identity_preconditioner,
        states: Iterable[VariationalState] = None,
        shifts: Iterable[float] = None,
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient.
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no
                preconditioner is used and the bare gradient is passed to the optimizer.
            states: a set of previously determined states in a list.
            shifts: contains the penalty coefficient for each previously determined state.
        """
        hamiltonian = hamiltonian.collect()
        hamiltonian_with_penalty = OperatorWithPenalty(hamiltonian, states, shifts)

        super().__init__(
            hamiltonian_with_penalty,
            optimizer,
            variational_state=variational_state,
            preconditioner=preconditioner,
        )

        self._raw_hamiltonian = hamiltonian

    @property
    def states(self) -> Iterable[VariationalState]:
        "..."
        return self._ham.states

    @property
    def shifts(self) -> Iterable[float]:
        """..."""
        return self._ham.shifts

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self.state.reset()

        # we need to reset MCMC samples for each penalty state in the list.
        for state_i in self._ham.states:
            state_i.reset()

        # Compute the local energy estimator and average Energy.
        (
            self._loss_stats,
            self._constraint_loss_stats,
        ), self._loss_grad = self.state.expect_and_grad(
            self._ham,
        )

        # if it's the identity it does
        # self._dp = self._loss_grad
        self._dp = self.preconditioner(self.state, self._loss_grad)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = jax.tree_map(
            lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
            self._dp,
            self.state.parameters,
        )

        return self._dp

    @property
    def energy(self) -> Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    @property
    def loss(self) -> Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_constraint_loss_stats_stats

    def __repr__(self):
        return (
            "VmcWithPenalty("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Hamiltonian    ", self._ham),
                ("Optimizer      ", self._optimizer),
                ("Preconditioner ", self.preconditioner),
                ("State          ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)
