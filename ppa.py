import netket as nk
import numpy as np

import netket.experimental as nkx
import flax.linen as nn
import jax.numpy as jnp
import jax

from functools import partial

# 1D chain
L = 10

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=0.5)

# RBM Spin Machine
# ma = nk.models.RBM(alpha=1, use_visible_bias=True, param_dtype=complex)


class Jastrow(nn.Module):
    N: int

    def setup(self):
        # self.symmetric_params_r = self.param('symmetric_params_r', nn.initializers.normal(dtype=jnp.float64), (self.N * (self.N - 1) // 2,))
        # self.symmetric_params_i = self.param('symmetric_params_i', nn.initializers.normal(dtype=jnp.float64), (self.N * (self.N - 1) // 2,))
        self.Jr = self.param(
            "Jr", nn.initializers.normal(dtype=jnp.float64), (self.N, self.N)
        )
        self.Ji = self.param(
            "Ji", nn.initializers.normal(dtype=jnp.float64), (self.N, self.N)
        )

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)

        # We create the parameter v, which is a vector of length N_sites
        v_bias_r = self.param(
            "visible_bias_r", nn.initializers.normal(dtype=jnp.float64), (self.N,)
        )
        v_bias_i = self.param(
            "visible_bias_i", nn.initializers.normal(dtype=jnp.float64), (self.N,)
        )
        J = self.Jr + 1j * self.Ji
        J = J + J.conj().T

        v_bias = v_bias_r + 1.0j * v_bias_i

        # Reshape the 1D array to a 2D symmetric matrix
        # J = self.reshape_symmetric_matrix(self.symmetric_params_r,self.N)+1.0j*self.reshape_symmetric_matrix(self.symmetric_params_i,self.N)

        # # Pass the input through the CNN
        x = OneDConvNet()(x) + x

        y = jnp.einsum("...i,ij,...j", x, J, x)

        return y + x @ v_bias

    @staticmethod
    def reshape_symmetric_matrix(params, N):

        matrix = jnp.zeros((N, N), dtype=jnp.float64)
        tril_indices = jnp.tril_indices(N, -1)
        matrix = matrix.at[tril_indices].set(params)
        matrix = matrix + matrix.T
        return matrix


class OneDConvNet(nn.Module):
    channels: int = 3
    kernel_size: tuple = (6,)
    activation: int = nn.tanh

    @nn.compact
    def __call__(self, x):
        batch_dims = x.shape[:-1]
        N = x.shape[-1]
        # reshape as (batch, sites, 1) where 1 is channel dim
        x = x.reshape(-1, N, 1)

        conv = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel_size,
            padding="CIRCULAR",
            param_dtype=jnp.float64,
        )

        x = conv(x)
        x = self.activation(x)

        x = nn.Dense(features=1, param_dtype=jnp.float64)(x)

        # restore original batch dimensions
        x = x.reshape(*batch_dims, N)
        return x


ma = Jastrow(L)
# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=4048 * 2, n_discard_per_chain=16)

# Optimizer
op = nk.optimizer.Sgd(0.05)
sr = nk.optimizer.SR(diag_shift=1e-4)

# Variational monte carlo driver
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Create observable
Sx = sum([nk.operator.spin.sigmax(hi, i) for i in range(L)]) / float(L)

# Run the optimization for 300 iterations to determine the ground state, used as
# initial state of the time-evolution
gs.run(n_iter=300, out="example_ising1d_GS", obs={"Sx": Sx})

# Create integrator for time propagation
integrator = nkx.dynamics.RK23(dt=0.01, adaptive=False, rtol=1e-3, atol=1e-3)
integrator = nkx.dynamics.Euler(dt=0.01)
print(integrator)

# Quenched hamiltonian: this has a different transverse field than `ha`
ha1 = nk.operator.Ising(hilbert=hi, graph=g, h=2)
te = nkx.TDVP(
    ha1,
    variational_state=vs,
    integrator=integrator,
    t0=0.0,
    qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=False, diag_shift=0.0),
    linear_solver=partial(nk.optimizer.solver.svd, rcond=1e-5),
    # qgt=nk.optimizer.qgt.QGTOnTheFly(diag_shift=1e-5)
    # linear_solver=jax.scipy.sparse.linalg.cg,
    # error_norm="qgt",
)

log = nk.logging.JsonLog("example_ising1d_TE")

# perform the time-evolution saving the observable Sx at every `tstop` time
te.run(
    T=1.0,
    out=log,
    show_progress=True,
    obs={"Sx": Sx, "Energy": ha1},
    tstops=np.linspace(0.0, 1.0, 101, endpoint=True),
)
