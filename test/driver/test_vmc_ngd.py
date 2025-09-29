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

from netket.models import RBM, RBMModPhase

from test.common import skipif_distributed

machines = [
    pytest.param(
        RBM(
            param_dtype=jnp.complex128,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            hidden_bias_init=jax.nn.initializers.normal(stddev=0.02),
            use_visible_bias=False,
        ),
        id="RBM(complex128)",
    ),
    pytest.param(RBMModPhase(), id="RBMModPhase"),
]
onthefly_vals = [pytest.param(True, id="OnTheFly"), pytest.param(False, id="Jacobian")]
ntk_onthefly_vals = [
    # pytest.param(False, True, id="SR+OnTheFly", marks=pytest.mark.skip(reason="Not Implemented")),
    pytest.param(False, False, id="SR+Jacobian"),
    pytest.param(True, True, id="NTK+OnTheFly"),
    pytest.param(True, False, id="NTK+Jacobian"),
]


class RBM(nn.Module):
    num_hidden: int  # Number of hidden neurons
    complex: bool
    real_output: bool = False

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
                use_bias=False,
                param_dtype=jnp.float64,
                kernel_init=jax.nn.initializers.normal(stddev=0.02),
                bias_init=jax.nn.initializers.normal(stddev=0.02),
            )

    def __call__(self, x):
        x = self.linearR(x)

        if self.complex:
            x = x + 1j * self.linearI(x)

        x = jnp.log(jax.lax.cosh(x))

        if self.real_output:
            return jnp.sum(x, axis=-1)
        elif self.complex:
            return jnp.sum(x, axis=-1)
        else:
            return jnp.sum(x, axis=-1).astype(jnp.complex128)


def _setup(*, complex=True, machine=None, real_output=False, chunk_size=None):
    L = 5
    Ns = L * L
    lattice = nk.graph.Square(L, max_neighbor_order=2)
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, inverted_ordering=False)
    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0)
    if nk.config.netket_experimental_sharding:
        H = H.to_jax_operator()

    # Define a variational state
    if machine is None:
        machine = RBM(num_hidden=Ns, complex=complex, real_output=real_output)

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
        chunk_size=chunk_size,
    )

    return H, opt, vstate


@pytest.mark.parametrize("model", machines)
@pytest.mark.parametrize("use_ntk, onthefly", ntk_onthefly_vals)
def test_advd_vs_nk_vmc(model, use_ntk, onthefly):
    """
    Checks that the advanced drivers VMC_SR gives the same results as the NetKet VMC with SR
    """
    n_iters = 5

    H, opt, vstate_srt = _setup(machine=model)
    gs = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vstate_srt,
        diag_shift=0.1,
        use_ntk=use_ntk,
        on_the_fly=onthefly,
        linear_solver=nk.optimizer.solver.cholesky,
    )
    logger_srt = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_srt)

    H, opt, vstate_sr = _setup(machine=model)
    gs = nk.driver.VMC(
        H,
        opt,
        variational_state=vstate_sr,
        preconditioner=nk.optimizer.SR(
            diag_shift=0.1,
            solver=nk.optimizer.solver.cholesky,
            qgt=nk.optimizer.qgt.QGTJacobianDense,
        ),
    )
    logger_sr = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_sr)

    # check same parameters
    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_srt.parameters, vstate_sr.parameters
    )

    if jax.process_index() == 0:
        energy_kernelSR = logger_srt.data["Energy"]["Mean"]
        energy_SR = logger_sr.data["Energy"]["Mean"]

        np.testing.assert_allclose(energy_kernelSR, energy_SR, atol=1e-10)


@pytest.mark.parametrize("model", machines)
@pytest.mark.parametrize("onthefly", onthefly_vals)
def test_SRt_vs_SR(model, onthefly):
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics as nk.driver.VMC with nk.optimizer.SR
    """

    n_iters = 5

    H, opt, vstate_srt = _setup(machine=model)
    gs = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vstate_srt,
        diag_shift=0.1,
        # proj_reg=0.5,
        # momentum=0.0,
        use_ntk=True,
        on_the_fly=onthefly,
    )
    logger_srt = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_srt)

    H, opt, vstate_sr = _setup(machine=model)
    gs = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vstate_sr,
        diag_shift=0.1,
        use_ntk=False,
        on_the_fly=False,
    )
    logger_sr = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_sr)

    # check same parameters
    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_srt.parameters, vstate_sr.parameters
    )

    if jax.process_index() == 0:
        energy_kernelSR = logger_srt.data["Energy"]["Mean"]
        energy_SR = logger_sr.data["Energy"]["Mean"]

        np.testing.assert_allclose(energy_kernelSR, energy_SR, atol=1e-10)


@skipif_distributed
@pytest.mark.parametrize("onthefly", onthefly_vals)
def test_SRt_real_vs_complex(onthefly):
    """s
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics for a positive definite wave function if jacobian_mode=complex or real
    """
    n_iters = 5

    H, opt, vstate_complex = _setup(complex=False)
    gs = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vstate_complex,
        diag_shift=0.1,
        # jacobian_mode="complex",
        # proj_reg=0.5,
        # momentum=0.0,
        use_ntk=True,
        on_the_fly=onthefly,
        mode="complex",
    )
    logger_complex = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_complex)

    H, opt, vstate_real = _setup(complex=False, real_output=True)
    gs = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vstate_real,
        diag_shift=0.1,
        # jacobian_mode="real",
        # proj_reg=0.5,
        # momentum=0.0,
        use_ntk=True,
        on_the_fly=onthefly,
        mode="real",
    )
    logger_real = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_real)

    # check same parameters
    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_complex.parameters, vstate_real.parameters
    )

    if jax.process_index() == 0:
        energy_complex = logger_complex.data["Energy"]["Mean"]
        energy_real = logger_real.data["Energy"]["Mean"]

        np.testing.assert_allclose(energy_real, energy_complex, atol=1e-10)


@skipif_distributed
def test_SRt_constructor_errors():
    """
    Error if jacobian mode is malformed.
    """
    H, opt, vstate_srt = _setup()
    gs = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vstate_srt,
        diag_shift=0.1,
        mode="complex",
    )
    assert gs.mode == "complex"
    gs.run(1)

    with pytest.raises(ValueError):
        gs = nk.driver.VMC_SR(
            H, opt, variational_state=vstate_srt, diag_shift=0.1, mode="belin"
        )


@skipif_distributed
def test_SRt_schedules():
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics as nk.driver.VMC with nk.optimizer.SR
    """
    H, opt, vstate_srt = _setup()
    gs = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vstate_srt,
        diag_shift=optax.linear_schedule(0.1, 0.001, 100),
    )
    gs.run(1)


@skipif_distributed
@pytest.mark.parametrize("use_ntk", [True, False])
def test_SRt_supports_netket_solvers(use_ntk):
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics as nk.driver.VMC with nk.optimizer.SR
    """
    H, opt, vstate_srt = _setup()
    gs = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vstate_srt,
        diag_shift=optax.linear_schedule(0.1, 0.001, 100),
        linear_solver=nk.optimizer.solver.pinv_smooth,
        use_ntk=use_ntk,
    )
    gs.run(1)


@pytest.mark.parametrize(
    "momentum", [pytest.param(None, id=""), pytest.param(0.9, id="momentum")]
)
@pytest.mark.parametrize("model", machines)
def test_srt_vs_ntk(model, momentum):
    """
    All nk.driver.VMC_kernelSR must give **exactly** the same dynamics even with momentum
    """
    n_iters = 5

    H, opt, vstate_srt = _setup(machine=model)
    gs = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vstate_srt,
        diag_shift=0.1,
        proj_reg=1.0,
        momentum=momentum,
        on_the_fly=False,
        use_ntk=True,
    )
    _, _, vstate_ntk = _setup(machine=model)
    gs_ntk = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vstate_ntk,
        diag_shift=0.1,
        proj_reg=1.0,
        momentum=momentum,
        on_the_fly=False,
        use_ntk=True,
    )
    logger_srt = nk.logging.RuntimeLog()
    logger_ntk = nk.logging.RuntimeLog()

    gs.run(n_iter=n_iters, out=logger_srt)
    gs_ntk.run(n_iter=n_iters, out=logger_ntk)

    # check same parameters
    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_srt.parameters, vstate_ntk.parameters
    )

    if jax.process_index() == 0:
        energy_kernelSR = logger_srt.data["Energy"]["Mean"]
        energy_SR = logger_ntk.data["Energy"]["Mean"]

        np.testing.assert_allclose(energy_kernelSR, energy_SR, atol=1e-10)


# WARNING: We spotted an instability of this code.
# If we run the test test_SRt_chunked for 500 iterations, the test fails when
# using momentum and comparing chunking vs no chunking. We are unsure about the origin of
# the instability. Most likely cause is numerical errors accumulating.


@pytest.mark.parametrize(
    "proj_reg", [pytest.param(None, id=""), pytest.param(1.0, id="proj_reg")]
)
@pytest.mark.parametrize(
    "momentum", [pytest.param(None, id=""), pytest.param(0.9, id="momentum")]
)
@pytest.mark.parametrize("model", machines)
@pytest.mark.parametrize("use_ntk, onthefly", ntk_onthefly_vals)
def test_SRt_chunked(use_ntk, onthefly, model, momentum, proj_reg):
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics with and without chunking
    """
    if not use_ntk:
        if momentum is not None or proj_reg is not None:
            pytest.skip("not implemented")

    n_iters = 5
    diag_shift = 0.01
    chunk_size = 64

    H, opt, vstate = _setup(machine=model, chunk_size=None)
    gs = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vstate,
        diag_shift=diag_shift,
        proj_reg=proj_reg,
        momentum=momentum,
        use_ntk=use_ntk,
        on_the_fly=onthefly,
    )
    logger = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger)

    _, _, vstate_chunked = _setup(machine=model, chunk_size=chunk_size)
    gs_chunked = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vstate_chunked,
        diag_shift=diag_shift,
        proj_reg=proj_reg,
        momentum=momentum,
        use_ntk=use_ntk,
        on_the_fly=onthefly,
    )
    gs_chunked.chunk_size_bwd = chunk_size

    logger_chunked = nk.logging.RuntimeLog()
    gs_chunked.run(n_iter=n_iters, out=logger_chunked)

    # check same parameters
    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate.parameters, vstate_chunked.parameters
    )

    if jax.process_index() == 0:
        energy = logger.data["Energy"]["Mean"]
        energy_chunked = logger_chunked.data["Energy"]["Mean"]

        np.testing.assert_allclose(energy, energy_chunked, atol=1e-10)
