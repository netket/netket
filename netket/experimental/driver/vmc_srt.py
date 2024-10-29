# Copyright 2023  The NetKet Authors - All rights reserved.
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

from collections.abc import Callable

from functools import partial
from textwrap import dedent
import warnings

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from netket import jax as nkjax
from netket import stats as nkstats
from netket.driver import AbstractVariationalDriver
from netket.errors import UnoptimalSRtWarning
from netket.jax import sharding
from netket.operator import AbstractOperator
from netket.utils import mpi, timing
from netket.utils.types import ScalarOrSchedule, Optimizer, PyTree
from netket.vqs import MCState

from jax.flatten_util import ravel_pytree


@timing.timed
@partial(jax.jit, static_argnames=("mode", "solver_fn"))
def SRt(
    O_L, local_energies, diag_shift, *, mode, solver_fn, e_mean=None, params_structure
):
    """
    For more details, see `https://arxiv.org/abs/2310.05715'. In particular,
    the following parallel implementation is described in Appendix "Distributed SR computation".
    """
    N_params = O_L.shape[-1]
    N_mc = O_L.shape[0] * mpi.n_nodes

    local_energies = local_energies.flatten()

    if e_mean is None:
        e_mean = nkstats.mean(local_energies)
    de = jnp.conj(local_energies - e_mean).squeeze()

    # * in this case O_L should be padded with zeros
    assert (N_params % mpi.n_nodes) == 0

    O_L = O_L / N_mc**0.5
    dv = -2.0 * de / N_mc**0.5

    if mode == "complex":
        # Concatenate the real and imaginary derivatives of the ansatz
        # O_L = jnp.concatenate((O_L[:, 0], O_L[:, 1]), axis=0)
        O_L = jnp.transpose(O_L, (1, 0, 2)).reshape(-1, O_L.shape[-1])

        dv = jnp.concatenate((jnp.real(dv), -jnp.imag(dv)), axis=-1)
    elif mode == "real":
        dv = dv.real
    else:
        raise NotImplementedError()

    # twons, (np, n_nodes) -> twons, np, n_nodes
    O_LT = O_L.reshape(O_L.shape[0], -1, mpi.n_nodes)
    # twons, np, n_nodes -> n_nodes, twons, np
    O_LT = jnp.moveaxis(O_LT, -1, 0)

    dv, token = mpi.mpi_gather_jax(dv)
    dv = dv.reshape(-1, *dv.shape[2:])
    O_LT, token = mpi.mpi_alltoall_jax(O_LT, token=token)

    # proc, twons, np -> (proc, twons) np
    O_LT = O_LT.reshape(-1, O_LT.shape[-1])

    matrix, token = mpi.mpi_reduce_sum_jax(O_LT @ O_LT.T, token=token)
    matrix_side = matrix.shape[-1]  # * it can be Ns or 2*Ns, depending on mode

    if mpi.rank == 0:
        matrix = matrix + diag_shift * jnp.eye(
            matrix_side
        )  # * shift diagonal regularization
        aus_vector = solver_fn(matrix, dv)
        # some solvers return a tuple, some others do not.
        # We check and try to support both
        if isinstance(aus_vector, tuple):
            aus_vector, _ = aus_vector

        aus_vector = aus_vector.reshape(mpi.n_nodes, -1)
        aus_vector, token = mpi.mpi_scatter_jax(aus_vector, token=token)
    else:
        shape = jnp.zeros((int(matrix_side / mpi.n_nodes),), dtype=jnp.float64)
        aus_vector, token = mpi.mpi_scatter_jax(shape, token=token)

    updates = O_L.T @ aus_vector
    updates, token = mpi.mpi_allreduce_sum_jax(updates, token=token)

    # If complex mode and we have complex parameters, we need
    # To repack the real coefficients in order to get complex updates
    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        np = updates.shape[-1] // 2
        updates = updates[:np] + 1j * updates[np:]

    return -updates


inv_default_solver = lambda A, b: jnp.linalg.inv(A) @ b
linear_solver = lambda A, b: jsp.linalg.solve(A, b, assume_a="pos")


@jax.jit
def _flatten_samples(x):
    return x.reshape(-1, x.shape[-1])


class VMC_SRt(AbstractVariationalDriver):
    r"""
    Energy minimization using Variational Monte Carlo (VMC) and the kernel
    formulation of Stochastic Reconfiguration (SR). This approach lead to
    *exactly* the same parameter updates of the standard SR with a
    diagonal shift regularization. For this reason, it is equivalent to the standard
    nk.driver.VMC with the preconditioner nk.optimizer.SR(solver=netket.optimizer.solver.solvers.solve)).
    In the kernel SR framework, the updates of the parameters can be written as:

    .. math::
        \delta \theta = \tau X(X^TX + \lambda \mathbb{I}_{2M})^{-1} f,

    where :math:`X \in R^{P \times 2M}` is the concatenation of the real and imaginary part
    of the centered Jacobian, with P the number of parameters and M the number of samples.
    The vector f is the concatenation of the real and imaginary part of the centered local
    energy. Note that, to compute the updates, it is sufficient to invert an :math:`M\times M` matrix
    instead of a :math:`P\times P` one. As a consequence, this formulation is useful
    in the typical deep learning regime where :math:`P \gg M`.

    See `R.Rende, L.L.Viteritti, L.Bardone, F.Becca and S.Goldt <https://arxiv.org/abs/2310.05715>`_
    for a detailed description of the derivation. A similar result can be obtained by minimizing the
    Fubini-Study distance with a specific constrain, see `A.Chen and M.Heyl <https://arxiv.org/abs/2302.01941>`_
    for details.
    """

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        *,
        diag_shift: ScalarOrSchedule,
        linear_solver_fn: Callable[[jax.Array, jax.Array], jax.Array] = linear_solver,
        jacobian_mode: str | None = None,
        variational_state: MCState = None,
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the
                    bare energy gradient.
            diag_shift: The diagonal shift of the stochastic reconfiguration matrix.
                        Typical values are 1e-4 ÷ 1e-3. Can also be an optax schedule.
            hamiltonian: The Hamiltonian of the system.
            linear_solver_fn: Callable to solve the linear problem associated to the
                              updates of the parameters
            jacobian_mode: The mode used to compute the jacobian of the variational state. Can be `'real'`
                    or `'complex'` (defaults to the dtype of the output of the model).
            variational_state: The :class:`netket.vqs.MCState` to be optimised. Other
                variational states are not supported.
        """
        super().__init__(variational_state, optimizer, minimized_quantity_name="Energy")

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

        if self.state.n_parameters % sharding.device_count() != 0:
            raise NotImplementedError(
                f"""
                VMC_SRt requires a network with a number of parameters
                multiple of the number of MPI devices/ranks in use.

                You have a network with {self.state.n_parameters}, but
                there are {sharding.device_count()} MPI ranks in use.

                To fix this, either add some 'fake' parameters to your
                network, or change the number of MPI nodes, or contribute
                some padding logic to NetKet!
                """
            )

        if self.state.n_parameters < self.state.n_samples:
            warnings.warn(
                UnoptimalSRtWarning(self.state.n_parameters, self.state.n_samples),
                UserWarning,
                stacklevel=2,
            )

        self._ham = hamiltonian.collect()  # type: AbstractOperator
        self._dp: PyTree = None

        self.diag_shift = diag_shift
        self.jacobian_mode = jacobian_mode
        self._linear_solver_fn = linear_solver_fn

        self._params_structure = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state.parameters
        )
        if not nkjax.tree_ishomogeneous(self._params_structure):
            raise ValueError(
                "SRt only supports neural networks with all real or all complex "
                "parameters. Hybrid structures are not yet supported (but we would welcome "
                "contributions. Get in touch with us!)"
            )

        _, unravel_params_fn = ravel_pytree(self.state.parameters)
        self._unravel_params_fn = jax.jit(unravel_params_fn)

    @property
    def jacobian_mode(self) -> str:
        """
        The mode used to compute the jacobian of the variational state. Can be `'real'`
        or `'complex'`.

        Real mode truncates imaginary part of the wavefunction, while `complex` does not.
        This internally uses :func:`netket.jax.jacobian`. See that function for a more
        complete documentation.
        """
        return self._jacobian_mode

    @jacobian_mode.setter
    def jacobian_mode(self, mode: str | None):
        if mode is None:
            mode = nkjax.jacobian_default_mode(
                self.state._apply_fun,
                self.state.parameters,
                self.state.model_state,
                self.state.samples,
                warn=False,
            )

        if mode not in ["complex", "real"]:
            raise ValueError(
                "`jacobian_mode` only supports 'real' for real-valued wavefunctions and"
                "'complex'.\n\n"
                "`holomorphic` is not yet supported, but could be contributed in the future."
            )
        self._jacobian_mode = mode

    @timing.timed
    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self.state.reset()

        # Compute the local energy estimator and average Energy
        local_energies = self.state.local_estimators(self._ham)

        self._loss_stats = nkstats.statistics(local_energies)

        samples = _flatten_samples(self.state.samples)
        jacobians = nkjax.jacobian(
            self.state._apply_fun,
            self.state.parameters,
            samples,
            self.state.model_state,
            mode=self.jacobian_mode,
            dense=True,
            center=True,
            chunk_size=self.state.chunk_size,
        )  # jacobians is centered

        diag_shift = self.diag_shift
        if callable(self.diag_shift):
            diag_shift = diag_shift(self.step_count)

        updates = SRt(
            jacobians,
            local_energies,
            diag_shift,
            mode=self.jacobian_mode,
            solver_fn=self._linear_solver_fn,
            e_mean=self._loss_stats.Mean,
            params_structure=self._params_structure,
        )

        self._dp = self._unravel_params_fn(updates)

        return self._dp

    @property
    def energy(self) -> nkstats.Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    def __repr__(self):
        return (
            "Vmc_SRt("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )
