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

import jax
import jax.numpy as jnp

from textwrap import dedent

from netket.utils.types import PyTree, Array, Callable
from netket.operator import AbstractOperator
from netket.stats import Stats
from netket.vqs import MCState
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)
from netket.utils import warn_deprecation
from netket.optimizer import Sgd

from netket.driver import VMC

from ..optimizer.rgn.model_and_operator_statistics import centered_jacobian_and_mean, en_grad_and_rhessian
from netket.optimizer.qgt.qgt_jacobian_common import _choose_jacobian_mode
from ..optimizer.rgn.hessian_plus_qgt_pytree import Hessian_Plus_QGT_PyTree 

class VMC_RGN(VMC):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    # TODO docstring
    def __init__(
        self,
        hamiltonian: AbstractOperator,
        eps_schedule: Callable,
        diag_shift_schedule: Callable,
        mode: str = None,
        chunk_size: int = None,
        optimizer=Sgd(learning_rate=1),
        variational_state=None,
        preconditioner: PreconditionerT = None,
        sr: PreconditionerT = None,
        sr_restart: bool = None,
        **kwargs,
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient. In this case the bare gradient is already scaled
                so use SGD with a learning rate 1 for best performance
            eps_schedule: function that takes the time step and returns a value for epsilon
            diag_shift_schedule: function that takes the time step and returns a value for 
            the diagonal shift
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no
                preconditioner is used and the bare gradient is passed to the optimizer.
        """

        super().__init__(*args, **kwargs)
        self.eps_schedule = eps_schedule
        self.diag_shift_schedule = diag_shift_schedule
        self.mode = mode

    def _forward_and_backward(self):

        self.state.reset()

        con_samples, mels = hamiltonian.get_conn_padded(self.state.samples)
        con_samples = con_samples.squeeze()
        mels = mels.squeeze()
    
        def forward_fn(W, σ):
            return self.state._apply_fun({"params": W, **self.state.model_state}, σ)

        jac, jac_mean = centered_jacobian_and_mean(forward_fn,self.state.parameters,self.state.samples,self.mode,self.chunk_size)
        en, grad, rhessian = en_grad_and_rhessian(forward_fn,self.state.parameters,self.state.samples,con_samples,mels,self.mode,self.chunk_size)

        eps = self.eps_schedule(self.step_count)
        diag_shift = self.diag_shift_schedule(self.step_count)

        self._dp = Hessian_Plus_QGT_PyTree(
            jac,
            jac_mean,
            rhessian,
            grad,
            en,
            eps,
            diag_shift
        )

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = jax.tree_multimap(
            lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
            self._dp,
            self.state.parameters,
        )

        return self._dp


