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
import netket as nk
import jax.numpy as jnp


def minimum_distance(x, sdim):
    """Computes distances between particles using mimimum image convention"""
    n_particles = x.shape[0] // sdim
    x = x.reshape(-1, sdim)

    distances = (-x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :])[
        jnp.triu_indices(n_particles, 1)
    ]
    distances = jnp.remainder(distances + L / 2.0, L) - L / 2.0

    return jnp.linalg.norm(distances, axis=1)


def potential(x, sdim):
    """Compute Aziz potential for single sample x"""
    dis = minimum_distance(x, sdim)
    eps = 7.846373
    A = 0.544850 * 10**6
    alpha = 13.353384
    c6 = 1.37332412
    c8 = 0.4253785
    c10 = 0.178100
    D = 1.241314

    return jnp.sum(
        eps
        * (
            A * jnp.exp(-alpha * dis)
            - (c6 / dis**6 + c8 / dis**8 + c10 / dis**10)
            * jnp.where(dis < D, jnp.exp(-((D / dis - 1) ** 2)), 1.0)
        )
    )


N = 10
d = 0.3  # 1/Angstrom
rm = 2.9673  # Angstrom
L = N / (0.3 * rm)
hilb = nk.hilbert.Particle(N=N, L=(L,), pbc=True)
sab = nk.sampler.MetropolisGaussian(hilb, sigma=0.05, n_chains=16, n_sweeps=32)


ekin = nk.operator.KineticEnergy(hilb, mass=1.0)
pot = nk.operator.PotentialEnergy(hilb, lambda x: potential(x, 1))
ha = ekin + pot

model = nk.models.DeepSetRelDistance(
    hilbert=hilb,
    cusp_exponent=5,
    layers_phi=2,
    layers_rho=3,
    features_phi=(16, 16),
    features_rho=(16, 16, 1),
)
vs = nk.vqs.MCState(sab, model, n_samples=4096, n_discard_per_chain=128)

op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

gs = nk.VMC(ha, op, sab, variational_state=vs, preconditioner=sr)
gs.run(n_iter=1000, out="Helium_10_1d")
