# Copyright 2023 The NetKet Authors - All rights reserved.
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

import netket as nk
import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

import pytest

from netket.experimental.driver import VMC_SRt
from netket.optimizer.solver.solvers import solve
from netket.utils import mpi
from netket.errors import UnoptimalSRtWarning


class RBM(nn.Module):
    num_hidden: int  # Number of hidden neurons
    complex: bool

    def setup(self):
        self.linearR = nn.Dense(
            features=self.num_hidden,
            use_bias=True,
            param_dtype=jnp.float64,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            bias_init=jax.nn.initializers.normal(stddev=0.02),
        )
        if self.complex:
            self.linearI = nn.Dense(
                features=self.num_hidden,
                use_bias=True,
                param_dtype=jnp.float64,
                kernel_init=jax.nn.initializers.normal(stddev=0.02),
                bias_init=jax.nn.initializers.normal(stddev=0.02),
            )

    def __call__(self, x):
        x = self.linearR(x)

        if self.complex:
            x = x + 1j * self.linearI(x)

        x = jnp.log(jax.lax.cosh(x))

        if self.complex:
            return jnp.sum(x, axis=-1)
        else:
            return jnp.sum(x, axis=-1).astype(jnp.complex128)


def _setup(*, complex=True, machine=None):
    L = 4
    Ns = L * L
    lattice = nk.graph.Square(L, max_neighbor_order=2)
    hi = nk.hilbert.Spin(s=1 / 2, total_sz=0, N=lattice.n_nodes)
    H = nk.operator.Heisenberg(
        hilbert=hi, graph=lattice, J=[1.0, 0.0], sign_rule=[-1, 1]
    )
    if nk.config.netket_experimental_sharding:
        H = H.to_jax_operator()

    # Define a variational state
    if machine is None:
        machine = RBM(num_hidden=2 * Ns, complex=complex)

    sampler = nk.sampler.MetropolisExchange(
        hilbert=hi,
        n_chains=64,
        graph=lattice,
        d_max=2,
    )
    opt = nk.optimizer.Sgd(learning_rate=0.035)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=machine,
        n_samples=512,
        n_discard_per_chain=0,
        seed=0,
        sampler_seed=0,
    )

    return H, opt, vstate


def test_SRt_vs_linear_solver_complexpars():
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics as nk.driver.VMC with nk.optimizer.SR
    """
    n_iters = 5

    model = nk.models.RBM(
        param_dtype=jnp.complex128,
        kernel_init=jax.nn.initializers.normal(stddev=0.02),
        hidden_bias_init=jax.nn.initializers.normal(stddev=0.02),
        use_visible_bias=False,
    )

    H, opt, vstate_srt = _setup(machine=model)
    gs = VMC_SRt(
        H, opt, variational_state=vstate_srt, diag_shift=0.1, jacobian_mode="complex"
    )
    logger_srt = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_srt)

    H, opt, vstate_sr = _setup(machine=model)
    sr = nk.optimizer.SR(solver=solve, diag_shift=0.1, holomorphic=False)
    gs = nk.driver.VMC(H, opt, variational_state=vstate_sr, preconditioner=sr)
    logger_sr = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_sr)

    # check same parameters
    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_srt.parameters, vstate_sr.parameters
    )

    if mpi.rank == 0 and jax.process_count() == 0:
        energy_kernelSR = logger_srt.data["Energy"]["Mean"]
        energy_SR = logger_sr.data["Energy"]["Mean"]

        np.testing.assert_allclose(energy_kernelSR, energy_SR, atol=1e-10)


def test_SRt_vs_linear_solver():
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics as nk.driver.VMC with nk.optimizer.SR
    """
    n_iters = 5

    H, opt, vstate_srt = _setup()
    gs = VMC_SRt(
        H, opt, variational_state=vstate_srt, diag_shift=0.1, jacobian_mode="complex"
    )
    logger_srt = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_srt)

    H, opt, vstate_sr = _setup()
    sr = nk.optimizer.SR(solver=solve, diag_shift=0.1, holomorphic=False)
    gs = nk.driver.VMC(H, opt, variational_state=vstate_sr, preconditioner=sr)
    logger_sr = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_sr)

    # check same parameters
    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_srt.parameters, vstate_sr.parameters
    )

    if mpi.rank == 0 and jax.process_count() == 0:
        energy_kernelSR = logger_srt.data["Energy"]["Mean"]
        energy_SR = logger_sr.data["Energy"]["Mean"]

        np.testing.assert_allclose(energy_kernelSR, energy_SR, atol=1e-10)


def test_SRt_real_vs_complex():
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics for a positive definite wave function if jacobian_mode=complex or real
    """
    n_iters = 5

    H, opt, vstate_complex = _setup(complex=False)
    gs = VMC_SRt(
        H,
        opt,
        variational_state=vstate_complex,
        diag_shift=0.1,
        jacobian_mode="complex",
    )
    logger_complex = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_complex)

    H, opt, vstate_real = _setup(complex=False)
    gs = VMC_SRt(
        H, opt, variational_state=vstate_real, diag_shift=0.1, jacobian_mode="real"
    )
    logger_real = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_real)

    # check same parameters
    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_complex.parameters, vstate_real.parameters
    )

    if mpi.rank == 0 and jax.process_count() == 0:
        energy_complex = logger_complex.data["Energy"]["Mean"]
        energy_real = logger_real.data["Energy"]["Mean"]

        np.testing.assert_allclose(energy_real, energy_complex, atol=1e-10)


def test_SRt_constructor_errors():
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics as nk.driver.VMC with nk.optimizer.SR
    """
    H, opt, vstate_srt = _setup()
    gs = VMC_SRt(
        H,
        opt,
        variational_state=vstate_srt,
        diag_shift=0.1,
    )
    assert gs.jacobian_mode == "complex"
    gs.run(1)

    with pytest.raises(ValueError):
        gs = VMC_SRt(
            H, opt, variational_state=vstate_srt, diag_shift=0.1, jacobian_mode="belin"
        )


def test_SRt_constructor_warns():
    H, opt, vstate = _setup(complex=False)
    with pytest.warns(UnoptimalSRtWarning):
        # more than parameters
        vstate.n_samples = 1024
        assert vstate.n_samples > vstate.n_parameters
        _ = VMC_SRt(H, opt, variational_state=vstate, diag_shift=0.1)


def test_SRt_schedules():
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics as nk.driver.VMC with nk.optimizer.SR
    """
    H, opt, vstate_srt = _setup()
    gs = VMC_SRt(
        H,
        opt,
        variational_state=vstate_srt,
        diag_shift=optax.linear_schedule(0.1, 0.001, 100),
    )
    gs.run(5)


def test_SRt_supports_netket_solvers():
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics as nk.driver.VMC with nk.optimizer.SR
    """
    H, opt, vstate_srt = _setup()
    gs = VMC_SRt(
        H,
        opt,
        variational_state=vstate_srt,
        diag_shift=optax.linear_schedule(0.1, 0.001, 100),
        linear_solver_fn=nk.optimizer.solver.pinv_smooth,
    )
    gs.run(5)
