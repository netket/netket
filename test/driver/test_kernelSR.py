import netket as nk
import numpy as np
from netket.experimental.driver import VMC_kernelSR

from netket.optimizer.solver.solvers import solve

import jax
import jax.numpy as jnp
from flax import linen as nn


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


def _setup(complex=True):
    L = 4
    Ns = L * L
    lattice = nk.graph.Square(L, max_neighbor_order=2)
    hi = nk.hilbert.Spin(s=1 / 2, total_sz=0, N=lattice.n_nodes)
    H = nk.operator.Heisenberg(
        hilbert=hi, graph=lattice, J=[1.0, 0.0], sign_rule=[-1, 1]
    )

    # Define a variational state
    machine = RBM(num_hidden=2 * Ns, complex=complex)

    sampler = nk.sampler.MetropolisExchange(
        hilbert=hi,
        n_chains=2048,
        graph=lattice,
        d_max=2,
    )
    opt = nk.optimizer.Sgd(learning_rate=0.035)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=machine,
        n_samples=2048,
        n_discard_per_chain=0,
        seed=0,
        sampler_seed=0,
    )

    return H, opt, vstate


def test_kernelSR_vs_linear_solver():
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics as nk.driver.VMC with nk.optimizer.SR
    """
    H, opt, vstate = _setup()
    gs = VMC_kernelSR(
        H, opt, variational_state=vstate, diag_shift=0.1, jacobian_mode="complex"
    )
    logger = gs.run(n_iter=10, out="ground_state")
    energy_kernelSR = logger[0].data["Energy"]["value"]

    H, opt, vstate = _setup()
    sr = nk.optimizer.SR(solver=solve, diag_shift=0.1, holomorphic=False)
    gs = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)
    logger = gs.run(n_iter=10, out="ground_state")
    energy_SR = logger[0].data["Energy"]["Mean"]

    assert np.allclose(energy_kernelSR, energy_SR, atol=1e-10)


def test_kernelSR_real_vs_complex():
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics for a positive definite wave function if jacobian_mode=complex or real
    """
    H, opt, vstate = _setup(complex=False)
    gs = VMC_kernelSR(
        H, opt, variational_state=vstate, diag_shift=0.1, jacobian_mode="complex"
    )
    logger = gs.run(n_iter=10, out="ground_state")
    energy_complex = logger[0].data["Energy"]["value"]

    H, opt, vstate = _setup(complex=False)
    gs = VMC_kernelSR(
        H, opt, variational_state=vstate, diag_shift=0.1, jacobian_mode="real"
    )
    logger = gs.run(n_iter=10, out="ground_state")
    energy_real = logger[0].data["Energy"]["value"]

    assert np.allclose(energy_real, energy_complex, atol=1e-10)
