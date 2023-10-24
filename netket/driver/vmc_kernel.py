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

from typing import Optional
from einops import rearrange

import jax
import jax.numpy as jnp

from textwrap import dedent
from inspect import signature
import netket.jax as nkjax
from netket.utils.types import PyTree
from netket.operator import AbstractOperator
from netket.stats import Stats
from netket.vqs import MCState
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
    _DeprecatedPreconditionerSignature,
)

from .vmc_common import info
from .abstract_variational_driver import AbstractVariationalDriver
import mpi4jax
from mpi4py import MPI

from jax.flatten_util import ravel_pytree

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
comm_jax = comm.Clone()

MASTER = 0

@jax.jit
def kernel_SR_memorysave(O_L, de, diag_shift):
    N_mc = O_L.shape[0] * size
    O_L = jnp.concatenate((jnp.real(O_L), jnp.imag(O_L)), axis=0)
    O_L = O_L / N_mc**0.5

    dv = de / N_mc**0.5
    dv = jnp.concatenate((jnp.real(dv), -jnp.imag(dv)), axis=-1)
    dv = -2.0 * dv

    O_LT = rearrange(O_L, 'twons (np proc) -> proc twons np', proc=size)
    
    dv, token = mpi4jax.gather(dv, root=MASTER, comm=comm_jax)
    dv = dv.reshape(-1, *dv.shape[2:])
    O_LT, token = mpi4jax.alltoall(O_LT, comm=comm_jax, token=token)

    O_LT = rearrange(O_LT, 'proc twons np -> (proc twons) np')
    
    matrix, token = mpi4jax.reduce(O_LT@O_LT.T, op=MPI.SUM, root=MASTER, comm=comm_jax, token=token)
    
    if rank==MASTER:
        matrix = jnp.linalg.inv(matrix + diag_shift * jnp.eye(2*N_mc))
        aus_vector = matrix @ dv
        aus_vector = aus_vector.reshape(size, -1)
        aus_vector, token = mpi4jax.scatter(aus_vector, root=MASTER, comm=comm_jax, token=token)
    else:
        shape = jnp.zeros((int(2*N_mc/size),), dtype=jnp.float64)
        aus_vector, token = mpi4jax.scatter(shape, root=MASTER, comm=comm_jax, token=token)

    updates = O_L.T @ aus_vector
    updates, token = mpi4jax.allreduce(updates, op=MPI.SUM, comm=comm_jax, token=token)
    
    return updates 

class VMC_kernelSR(AbstractVariationalDriver):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer,
        *args,
        variational_state=None,
        preconditioner: PreconditionerT = identity_preconditioner,
        **kwargs,
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
        """
        if variational_state is None:
            variational_state = MCState(*args, **kwargs)

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

        _, self.unravel_params_fn = ravel_pytree(self.state.parameters)


        self.preconditioner = preconditioner
        self.diag_shift = self.preconditioner.diag_shift
        self._dp: PyTree = None
        self._S = None
        self._sr_info = None

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

        Often, this is taken to be :func:`nk.optimizer.SR`. If it is set to
        `None`, then the identity is used.
        """
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, val: Optional[PreconditionerT]):
        if val is None:
            val = identity_preconditioner

        if len(signature(val).parameters) == 2:
            val = _DeprecatedPreconditionerSignature(val)

        self._preconditioner = val

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self.state.reset()

        # Compute the local energy estimator and average Energy
        local_energy = self.state.local_estimators(self._ham).squeeze()
        local_energy = local_energy / 4
        e_mean = local_energy.mean()
        self._loss_stats = e_mean / self.state._samples.shape[-1]

        de = jnp.conj(local_energy - e_mean)

        jacobians = nkjax.jacobian( self.state._apply_fun,
                                    self.state.parameters,
                                    self.state._samples.squeeze(), #! THIS IS NOT CORRECT ...
                                    self.state.model_state,
                                    mode="complex",
                                    dense=True,
                                    center=False,
                                    )
        
        O_L = jacobians[:, 0] + 1j * jacobians[:, 1]
        O_L = O_L - jnp.mean(O_L, axis=0, keepdims=True)

        updates = kernel_SR_memorysave(O_L, de, self.diag_shift)

        self._dp = self.unravel_params_fn(-updates)

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

    def info(self, depth=0):
        lines = [
            f"{name}: {info(obj, depth=depth + 1)}"
            for name, obj in [
                ("Hamiltonian    ", self._ham),
                ("Optimizer      ", self._optimizer),
                ("Preconditioner ", self.preconditioner),
                ("State          ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)
