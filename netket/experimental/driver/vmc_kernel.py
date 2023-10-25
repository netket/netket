from functools import partial
from typing import Callable
from einops import rearrange

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from netket.driver.vmc import VMC

import netket.jax as nkjax
from netket.operator import AbstractOperator

from netket.utils.mpi import rank, n_nodes
from netket.utils.mpi import mpi_gather_jax, mpi_alltoall_jax, mpi_reduce_sum_jax, mpi_scatter_jax, mpi_allreduce_sum_jax

from jax.flatten_util import ravel_pytree


@partial(jax.jit, static_argnums=(3,4))
def kernel_SR(O_L, de, diag_shift, mode, solver_fn):
    """
    For more details, see `https://arxiv.org/abs/2310.05715'. In particular, 
    the following parallel implementation is described in Appendix "Distributed SR computation".
    """
    N_params = O_L.shape[-1]
    N_mc = O_L.shape[0] * n_nodes

    if N_params % n_nodes != 0:
        raise NotImplementedError() #* in this case O_L should be padded with zeros

    O_L = O_L / N_mc**0.5
    dv = -2.0 * de / N_mc**0.5

    if mode=="complex":
        O_L = jnp.concatenate((O_L[:, 0], O_L[:, 1]), axis=0)
        dv = jnp.concatenate((jnp.real(dv), -jnp.imag(dv)), axis=-1)
    elif mode=="real":
        dv = dv.real
    else:
        raise NotImplementedError()

    O_LT = rearrange(O_L, 'twons (np proc) -> proc twons np', proc=n_nodes)
    
    dv, token = mpi_gather_jax(dv)
    dv = dv.reshape(-1, *dv.shape[2:])
    O_LT, token = mpi_alltoall_jax(O_LT, token=token)

    O_LT = rearrange(O_LT, 'proc twons np -> (proc twons) np')
    
    matrix, token = mpi_reduce_sum_jax(O_LT@O_LT.T, token=token)
    matrix_side = matrix.shape[-1] #* it can be Ns or 2*Ns, depending on mode

    if rank==0:
        matrix = matrix + diag_shift * jnp.eye(matrix_side) #* shift diagonal regularization
        aus_vector = solver_fn(matrix, dv)
        aus_vector = aus_vector.reshape(n_nodes, -1)
        aus_vector, token = mpi_scatter_jax(aus_vector, token=token)
    else:
        shape = jnp.zeros((int(matrix_side/n_nodes),), dtype=jnp.float64)
        aus_vector, token = mpi_scatter_jax(shape, token=token)

    updates = O_L.T @ aus_vector
    updates, token = mpi_allreduce_sum_jax(updates, token=token)
    
    return -updates 

inv_default_solver = lambda A, b: jnp.linalg.inv(A) @ b
linear_solver = lambda A, b: jsp.linalg.solve(A, b, assume_a="pos")

class VMC_kernelSR(VMC):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer,
        diag_shift,
        *args,
        linear_solver_fn: Callable[[jax.Array, jax.Array], jax.Array] = linear_solver,
        jacobian_mode = None,
        variational_state = None,
        **kwargs,
    ):
        super().__init__(hamiltonian, optimizer, variational_state=variational_state)

        self._ham = hamiltonian.collect()  # type: AbstractOperator
        _, self.unravel_params_fn = ravel_pytree(self.state.parameters)
        self.diag_shift = diag_shift
        self.jacobian_mode = jacobian_mode
        self.linear_solver_fn = linear_solver_fn

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self.state.reset()

        # Compute the local energy estimator and average Energy
        local_energy = self.state.local_estimators(self._ham).squeeze()
        
        e_mean = local_energy.mean()
        self._loss_stats = e_mean

        de = jnp.conj(local_energy - e_mean)

        if self.jacobian_mode is None:
            mode = "complex"
        else:
            #* mode='complex' is the most general; mode='holomorphic' could be implemented
            assert self.jacobian_mode in ["complex", "real"], "Jacobian mode must be 'complex' or 'real'"
            mode = self.jacobian_mode

        jacobians = nkjax.jacobian(self.state._apply_fun,
                                self.state.parameters,
                                self.samples.squeeze(), 
                                self.state.model_state,
                                mode=mode,
                                dense=True,
                                center=True) #* jaxcobians is centered

        updates = kernel_SR(jacobians, de, self.diag_shift, mode, self.linear_solver_fn)

        self._dp = self.unravel_params_fn(updates)

        return self._dp
    
    @property
    def samples(self):
        return self.state._samples #! is there a better way?