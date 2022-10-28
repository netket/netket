# Copyright 2020, 2021  The NetKet Authors - All rights reserved.
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

from typing import Callable, Union
from functools import partial

import jax
import jax.numpy as jnp

import netket as nk
from netket.driver.vmc_common import info
from netket.operator import AbstractOperator
from netket.optimizer import LinearOperator
from netket.optimizer.qgt import QGTAuto
from netket.vqs import VariationalState, VariationalMixedState, MCState, ExactState

from netket.experimental.dynamics import RKIntegratorConfig

from .tdvp_common import TDVPBaseDriver, odefun


class TDVP(TDVPBaseDriver):
    """
    Variational time evolution based on the time-dependent variational principle which,
    when used with Monte Carlo sampling via :class:`netket.vqs.MCState`, is the time-dependent VMC
    (t-VMC) method.

    .. note::
        This TDVP Driver uses the time-integrators from the `nkx.dynamics` module, which are
        automatically executed under a `jax.jit` context.

        When running computations on GPU, this can lead to infinite hangs or extremely long
        compilation times. In those cases, you might try setting the configuration variable
        `nk.config.netket_experimental_disable_ode_jit = True` to mitigate those issues.

    """

    def __init__(
        self,
        operator: AbstractOperator,
        variational_state: VariationalState,
        integrator: RKIntegratorConfig,
        *,
        t0: float = 0.0,
        propagation_type="real",
        qgt: LinearOperator = None,
        linear_solver=nk.optimizer.solver.svd,
        linear_solver_restart: bool = False,
        error_norm: Union[str, Callable] = "euclidean",
    ):
        r"""
        Initializes the time evolution driver.

        Args:
            operator: The generator of the dynamics (Hamiltonian for pure states,
                Lindbladian for density operators).
            variational_state: The variational state.
            integrator: Configuration of the algorithm used for solving the ODE.
            t0: Initial time at the start of the time evolution.
            propagation_type: Determines the equation of motion: "real"  for the
                real-time Schrödinger equation (SE), "imag" for the imaginary-time SE.
            qgt: The QGT specification.
            linear_solver: The solver for solving the linear system determining the time evolution.
                This must be a jax-jittable function :code:`f(A,b) -> x` that accepts a Matrix-like, Linear Operator
                PyTree object :math:`A` and a vector-like PyTree :math:`b` and returns the PyTree :math:`x` solving
                the system :math:`Ax=b`.
                Defaults to :func:`nk.optimizer.solver.svd` with the default svd threshold of 1e-10.
                To change the svd threshold you can use :func:`functools.partial` as follows:
                :code:`functools.partial(nk.optimizer.solver.svd, rcond=1e-4)`.
            linear_solver_restart: If False (default), the last solution of the linear system
                is used as initial value in subsequent steps.
            error_norm: Norm function used to calculate the error with adaptive integrators.
                Can be either "euclidean" for the standard L2 vector norm :math:`w^\dagger w`,
                "maximum" for the maximum norm :math:`\max_i |w_i|`
                or "qgt", in which case the scalar product induced by the QGT :math:`S` is used
                to compute the norm :math:`\Vert w \Vert^2_S = w^\dagger S w` as suggested
                in PRL 125, 100503 (2020).
                Additionally, it possible to pass a custom function with signature
                :code:`norm(x: PyTree) -> float`
                which maps a PyTree of parameters :code:`x` to the corresponding norm.
                Note that norm is used in jax.jit-compiled code.
        """
        if qgt is None:
            qgt = QGTAuto(solver=linear_solver)

        self.propagation_type = propagation_type
        if isinstance(variational_state, VariationalMixedState):
            # assuming Lindblad Dynamics
            # TODO: support density-matrix imaginary time evolution
            if propagation_type == "real":
                self._loss_grad_factor = 1.0
            else:
                raise ValueError(
                    "only real-time Lindblad evolution is supported for " "mixed states"
                )
        else:
            if propagation_type == "real":
                self._loss_grad_factor = -1.0j
            elif propagation_type == "imag":
                self._loss_grad_factor = -1.0
            else:
                raise ValueError("propagation_type must be one of 'real', 'imag'")

        self.qgt = qgt
        self.linear_solver = linear_solver
        self.linear_solver_restart = linear_solver_restart

        super().__init__(
            operator, variational_state, integrator, t0=t0, error_norm=error_norm
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("generator     ", self._generator_repr),
                ("integrator    ", self._integrator),
                ("linear solver ", self.linear_solver),
                ("state         ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)


@odefun.dispatch
def odefun_tdvp(  # noqa: F811
    state: Union[MCState, ExactState], driver: TDVP, t, w, *, stage=0
):
    # pylint: disable=protected-access

    state.parameters = w
    state.reset()

    op_t = driver.generator(t)

    driver._loss_stats, driver._loss_forces = state.expect_and_forces(
        op_t,
    )
    driver._loss_grad = _map_parameters(
        driver._loss_forces,
        state.parameters,
        driver._loss_grad_factor,
        driver.propagation_type,
        type(state),
    )

    qgt = driver.qgt(driver.state)
    if stage == 0:  # TODO: This does not work with FSAL.
        driver._last_qgt = qgt

    initial_dw = None if driver.linear_solver_restart else driver._dw
    driver._dw, _ = qgt.solve(driver.linear_solver, driver._loss_grad, x0=initial_dw)

    # If parameters are real, then take only real part of the gradient (if it's complex)
    driver._dw = jax.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
        driver._dw,
        state.parameters,
    )

    return driver._dw


@partial(jax.jit, static_argnums=(3, 4))
def _map_parameters(forces, parameters, loss_grad_factor, propagation_type, state_T):

    forces = jax.tree_map(
        lambda x, target: loss_grad_factor * x,
        forces,
        parameters,
    )

    forces = jax.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
        forces,
        parameters,
    )

    return forces
