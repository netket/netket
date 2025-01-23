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
from functools import partial
from textwrap import dedent
import warnings

import jax
import jax.numpy as jnp

from netket import jax as nkjax
from netket import stats as nkstats
from netket.driver import AbstractVariationalDriver
from netket.errors import UnoptimalSRtWarning
from netket.jax import sharding
from netket.operator import AbstractOperator
from netket.utils import mpi
from netket.utils.types import ScalarOrSchedule, Optimizer, PyTree
from netket.vqs import MCState

from netket.utils.optional_deps import import_optional_dependency

from einops import rearrange
from netket.jax.sharding import shard_along_axis, all_gather

nt = import_optional_dependency("neural_tangents", descr="ntk")


@partial(
    jax.jit,
    static_argnames=(
        "apply_fn",
        "nbatches",
    ),
)
def SR_ntk_complex(
    samples, params, apply_fn, model_state, local_energies, diag_shift, nbatches=1
):
    local_energies = local_energies.flatten()
    e_mean = nkstats.mean(local_energies)
    de = jnp.conj(local_energies - e_mean).squeeze()

    N_mc = de.size * mpi.n_nodes
    dv = -2.0 * de / N_mc**0.5

    dv = all_gather(dv)
    dv = jnp.concatenate((jnp.real(dv), -jnp.imag(dv)), axis=-1)  # shape [2*N_mc,]

    all_samples = all_gather(samples)

    def _apply_fn(params, samples):
        log_amp = apply_fn({"params": params, **model_state}, samples)
        re, im = log_amp.real, log_amp.imag
        return jnp.concatenate((re[:, None], im[:, None]), axis=-1)  # shape [N_mc,2]

    kwargs = dict(f=_apply_fn, trace_axes=(), vmap_axes=0)
    jacobian_contraction = nt.empirical_ntk_fn(
        **kwargs, implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION
    )

    if nbatches > 1:
        all_samples = jax.tree_map(
            lambda x: x.reshape(nbatches, -1, *x.shape[1:]), all_samples
        )
        aus_func = lambda batch_lattice: jacobian_contraction(
            samples, batch_lattice, params
        )
        ntk_local = jax.lax.map(aus_func, all_samples)
        ntk_local = rearrange(ntk_local, "nbatches i j z w -> i (nbatches j) z w")
    else:
        ntk_local = jacobian_contraction(
            samples, all_samples, params
        )  # shape [N_mc/p.size, N_mc, 2, 2]

    ntk = all_gather(ntk_local)  # shape [N_mc, N_mc, 2, 2]
    ntk = rearrange(
        ntk, "i j z w -> (z i) (w j)"
    )  # shape [2*N_mc, 2*N_mc] checked with direct calculation of J^T J

    delta = jnp.eye(N_mc) - 1 / N_mc  # shape [N_mc, N_mc] symmetric matrix
    delta_conc = jnp.zeros((2 * N_mc, 2 * N_mc))
    delta_conc = delta_conc.at[:N_mc, :N_mc].set(delta)
    delta_conc = delta_conc.at[N_mc : 2 * N_mc, N_mc : 2 * N_mc].set(
        delta
    )  # shape [2*N_mc, 2*N_mc]

    ntk = delta_conc @ ntk @ delta_conc  # shape [2*N_mc, 2*N_mc] centering the jacobian

    # add diagonal shift
    ntk = ntk / N_mc + diag_shift * jnp.eye(ntk.shape[0])
    aus_vector = jax.scipy.linalg.solve(ntk, dv, assume_a="sym")
    # aus_vector = jnp.linalg.inv(ntk) @ dv

    aus_vector = aus_vector / N_mc**0.5  # shape [2*N_mc,]

    aus_vector = delta_conc @ aus_vector

    aus_vector = aus_vector.reshape(2, -1).T  # shape [N_mc,2]
    aus_vector = shard_along_axis(aus_vector, axis=0)

    f = lambda params: _apply_fn(params, samples)
    _, vjp_fun = jax.vjp(f, params)
    updates = vjp_fun(aus_vector)[0]  # pytree [N_params,]

    updates, _ = mpi.mpi_allreduce_sum_jax(updates)

    updates = jax.tree_util.tree_map(lambda x: -x, updates)

    return updates


@jax.jit
def _flatten_samples(x):
    return x.reshape(-1, x.shape[-1])


class VMC_SRt_ntk(AbstractVariationalDriver):
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
        variational_state: MCState = None,
        chunk_size: int = None,
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the
                    bare energy gradient.
            diag_shift: The diagonal shift of the stochastic reconfiguration matrix.
                        Typical values are 1e-4 ÷ 1e-3. Can also be an optax schedule.
            variational_state: The :class:`netket.vqs.MCState` to be optimised. Other
                variational states are not supported.
            chunk_size: The size of the chunks in which the ntk is computed.
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

        self._params_structure = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state.parameters
        )
        if not nkjax.tree_ishomogeneous(self._params_structure):
            raise ValueError(
                "SRt only supports neural networks with all real or all complex "
                "parameters. Hybrid structures are not yet supported (but we would welcome "
                "contributions. Get in touch with us!)"
            )

        if chunk_size is None:
            self.nbatches = 1
        elif self.state.n_samples % chunk_size != 0:
            raise ValueError("Chunk size must divide number of samples!")
        else:
            self.nbatches = self.state.n_samples // chunk_size

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

        diag_shift = self.diag_shift
        if callable(self.diag_shift):
            diag_shift = diag_shift(self.step_count)

        updates = SR_ntk_complex(
            samples,
            self.state.parameters,
            self.state._apply_fun,
            self.state.model_state,
            local_energies,
            diag_shift,
            nbatches=self.nbatches,
        )

        self._dp = updates

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
