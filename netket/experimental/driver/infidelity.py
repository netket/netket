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


from textwrap import dedent
from typing import Optional
import flax
import jax

from netket.utils import timing
from netket.utils.types import PyTree, Optimizer
from netket.operator import AbstractOperator
from netket.stats import Stats
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)
from netket.vqs import VariationalState, MCState, FullSumState
from netket.jax import tree_cast

from netket.driver.abstract_variational_driver import AbstractVariationalDriver
from netket.experimental.operator.infidelity import InfidelityOperator
from netket.jax import HashablePartial


class InfidelityOptimization(AbstractVariationalDriver):
    """
    Infidelity minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self,
        target_state: VariationalState,
        optimizer: Optimizer,
        *,
        operator: AbstractOperator = None,
        variational_state: VariationalState,
        cv_coeff: Optional[float] = None,
        preconditioner: PreconditionerT = identity_preconditioner,
    ):
        r"""
        Constructs a driver that minimizes the infidelity between an input variational state and a target state.

        The target state can be defined in two ways:
        1. as a variational state that is passed as `target_state`.
        2. as a state obtained by applying an operator `U` to a variational state |ϕ⟩, i.e., |Φ⟩ = U|ϕ⟩.

        The operator I_op computing the infidelity I among two variational states math`|ψ⟩` and |Φ⟩ as:

        .. math::

        I = 1 - `math`|⟨ψ|Φ⟩|^2 / ⟨ψ|ψ⟩ ⟨Φ|Φ⟩ = 1 - ⟨ψ|I_op|ψ⟩ / ⟨ψ|ψ⟩

        where:

        .. math::

        I_op = |Φ⟩⟨Φ| / ⟨Φ|Φ⟩

        The Monte Carlo estimator of I is:

        ..math::

        I = \mathbb{E}_{χ}[ I_loc(σ,η) ] = \mathbb{E}_{χ}[ ⟨σ|Φ⟩ ⟨η|ψ⟩ / ⟨σ|ψ⟩ ⟨η|Φ⟩ ]

        where χ(σ, η) = |Ψ(σ)|^2 |Φ(η)|^2 / ⟨ψ|ψ⟩ ⟨Φ|Φ⟩ is the joint born distribution. This estimator
        can be utilized both when |Φ⟩ =|ϕ⟩ and when |Φ⟩ = U|ϕ⟩, with U a (unitary or
        non-unitary) operator. We remark that sampling from U|ϕ⟩ requires to compute connected
        elements of U and so is more expensive than sampling from an autonomous state.

        The gradient of I is calculated in the way following https://arxiv.org/pdf/2410.10720:
        ..math::
        \boldsymbol{\nabla} \mathcal{F} & =\boldsymbol{\nabla} \mathbb{E}_\chi[f(\sigma)]=\mathbb{E}_\chi[g(\sigma)]
        g(\sigma) & =2 \operatorname{Re}\{\Delta J(\sigma)\} f(\sigma)+\boldsymbol{\nabla} f(\sigma) .

        \chi(x, y)=\chi(x)=\pi_\psi(x)
        g(x, y)= 2\operatorname{Re}\left\{\Delta J(x) H_{\mathrm{loc}}(x)^*\right\} .

        Args:
        target_state: target variational state |ϕ⟩.
        optimizer: the optimizer to use (from optax)
        variational_state: the variational state to train
        U: operator U.
        U_dagger: dagger operator U^{\dagger}.
        cv_coeff: Control Variates coefficient c.
        dtype: The dtype of the output of expectation value and gradient.
        sample_Upsi: flag specifiying whether to sample from |ϕ⟩ or from U|ϕ⟩. If False with `is_unitary=False`, an error occurs.
        preconditioner: Determines which preconditioner to use for the loss gradient.
        This must be a tuple of `(object, solver)` as documented in the section
        `preconditioners` in the documentation. The standard preconditioner
        included with NetKet is Stochastic Reconfiguration. By default, no
        preconditioner is used and the bare gradient is passed to the optimizer.
        """

        if variational_state.hilbert != target_state.hilbert:
            raise TypeError(
                dedent(
                    f"""the variational_state has hilbert space {variational_state.hilbert}
                    (this is normally defined by the hilbert space in the sampler), but
                    the hamiltonian has hilbert space {target_state.hilbert}.
                    The two should match.
                    """
                )
            )

        super().__init__(
            variational_state, optimizer, minimized_quantity_name="Infidelity"
        )

        if operator is not None:

            def _logpsi_fun(apply_fun, variables, x, *args):
                variables_applyfun, O = flax.core.pop(variables, "operator")

                xp, mels = O.get_conn_padded(x)
                xp = xp.reshape(-1, x.shape[-1])
                logpsi_xp = apply_fun(variables_applyfun, xp, *args)
                logpsi_xp = logpsi_xp.reshape(mels.shape)

                return jax.scipy.special.logsumexp(logpsi_xp, axis=-1, b=mels)

            logpsi_fun = HashablePartial(_logpsi_fun, target_state._apply_fun)

            if isinstance(target_state, MCState):
                target_state = MCState(
                    sampler=target_state.sampler,
                    apply_fun=logpsi_fun,
                    n_samples=target_state.n_samples,
                    variables=flax.core.copy(
                        target_state.variables, {"operator": operator}
                    ),
                )
            if isinstance(target_state, FullSumState):
                target_state = FullSumState(
                    hilbert=target_state.hilbert,
                    apply_fun=logpsi_fun,
                    variables=flax.core.copy(
                        target_state.variables, {"operator": operator}
                    ),
                )

        self._target_state = target_state
        self.preconditioner = preconditioner
        self._infidelity_operator = InfidelityOperator(
            target_state=target_state, cv_coeff=cv_coeff
        )
        self._dp: PyTree = None
        self._S = None

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

    @timing.timed
    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self.state.reset()

        # Compute the local energy estimator and average Energy
        self._loss_stats, self._loss_grad = self.state.expect_and_grad(
            self._infidelity_operator
        )

        # if it's the identity it does
        # self._dp = self._loss_grad
        self._dp = self.preconditioner(self.state, self._loss_grad, self.step_count)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = tree_cast(self._dp, self.state.parameters)

        return self._dp

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
