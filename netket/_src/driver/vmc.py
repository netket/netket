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
from textwrap import dedent


from netket.utils import struct
from netket.utils.types import PyTree, Optimizer
from netket.operator import AbstractOperator
from netket.stats import Stats
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)
from netket.vqs import VariationalState
from netket.jax import tree_cast

from netket._src.driver.abstract_variational_driver import (
    AbstractVariationalDriver,
)


class VMC(AbstractVariationalDriver):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    _preconditioner: PreconditionerT = identity_preconditioner
    _ham: AbstractOperator = struct.field(pytree_node=False, serialize=False)
    _S: Any = struct.field(pytree_node=True, serialize=True)

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        *,
        variational_state: VariationalState,
        preconditioner: PreconditionerT = identity_preconditioner,
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient.
            variational_state: The variational state for which the hamiltonian must
                be minimised.
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no
                preconditioner is used and the bare gradient is passed to the optimizer.
        """
        if variational_state.hilbert != hamiltonian.hilbert:
            raise TypeError(
                dedent(
                    f"""the variational_state has hilbert space {variational_state.hilbert}
                    (this is normally defined by the hilbert space in the sampler), but
                    the hamiltonian has hilbert space {hamiltonian.hilbert}.
                    The two should match.
                    """
                )
            )

        super().__init__(variational_state, optimizer, minimized_quantity_name="Energy")

        self._ham = hamiltonian.collect()  # type: AbstractOperator

        self.preconditioner = preconditioner

        self._loss_stats = Stats()
        self._dp: PyTree = None
        self._S = None

    @property
    def preconditioner(self):
        """
        The preconditioner used to modify the gradient.

        This is a function with the following signature

        .. code-block:: python

            preconditioner(vstate: VariationalState,
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

    def compute_loss_and_update(self):

        # Compute the local energy estimator and average Energy
        energy, grad = self.state.expect_and_grad(self._ham)

        # if it is the identity it does
        # self._dp = self._loss_grad
        dp = self.preconditioner(self.state, grad, self.step_count)

        # If parameters are real, then take only real part of the gradient (if it is complex)
        dp = tree_cast(dp, self.state.parameters)

        return energy, dp

    @property
    def energy(self) -> Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    def __repr__(self):
        return (
            "Vmc("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )


# from netket.driver import VMC


# serialization
# def serialize_AbstractVariationalDriver(driver):
#     state_dict = {
#         "state": serialization.to_state_dict(driver._variational_state),
#         "optimizer_state": serialization.to_state_dict(driver._optimizer_state),
#         "loss_stats": serialization.to_state_dict(driver._loss_stats),
#         "step_count": driver._step_count,
#         # "timer": driver._timer,
#     }
#     return state_dict
#
#
# def deserialize_AbstractVariationalDriver(driver, state_dict):
#     import copy
#
#     new_driver = copy.copy(driver)
#     new_driver._variational_state = serialization.from_state_dict(
#         driver._variational_state, state_dict["state"]
#     )
#     new_driver._optimizer_state = serialization.from_state_dict(
#         driver._optimizer_state, state_dict["optimizer_state"]
#     )
#     new_driver._loss_stats = serialization.from_state_dict(
#         driver._loss_stats, state_dict["loss_stats"]
#     )
#     new_driver._step_count = state_dict["step_count"]
#
#     return new_driver
#
#
# # serialization.register_serialization_state(
# #     AbstractVariationalDriver,
# #     serialize_AbstractVariationalDriver,
# #     deserialize_AbstractVariationalDriver,
# # )
#
#
# # serialization
# def serialize_VMC(driver):
#     state_dict = serialize_AbstractVariationalDriver(driver)
#     state_dict["preconditioner"] = serialization.to_state_dict(driver.preconditioner)
#     return state_dict
#
#
# def deserialize_VMC(driver, state_dict):
#     driver = deserialize_AbstractVariationalDriver(driver, state_dict)
#     driver.preconditioner = serialization.from_state_dict(
#         driver.preconditioner, state_dict["preconditioner"]
#     )
#     return driver


# serialization.register_serialization_state(
#     VMC,
#     serialize_VMC,
#     deserialize_VMC,
# )
