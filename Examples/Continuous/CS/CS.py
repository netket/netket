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


def mycb(step, logged_data, driver):
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    return True


def minimum_distance(x, sdim):

    n_particles = x.shape[0] // sdim
    x = x.reshape(-1, sdim)

    distances = (-x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :])[
        jnp.triu_indices(n_particles, 1)
    ]
    return jnp.linalg.norm(distances, axis=1)


def potential(x):
    dis = minimum_distance(x, 1)

    dis = jnp.sin(jnp.pi / L * dis) ** 2
    beta = 2.0
    g = (jnp.pi / L) ** 2 * beta * (beta - 1)

    return jnp.sum(g / dis)


L = 15.0

hilb = nk.hilbert.Particle(N=10, L=(L,), pbc=True)

sab = nk.sampler.SingleMetropolisGaussian(hilb, sigma=1.0, n_chains=16)

model = nk.models.DeepSet(
    k=4,
    L=L,
    sdim=1,
    layers_phi=2,
    layers_rho=3,
    features_phi=(16, 16),
    features_rho=(16, 16, 1),
)
# model = nk.models.Gaussian()
ekin = nk.operator.KineticEnergy(hilb, mass=1.0)
pot = nk.operator.PotentialEnergy(hilb, potential)
ha = ekin + pot

vs = nk.vqs.MCState(sab, model, n_samples=10 ** 4, n_discard_per_chain=2000)
"""
import flax
with open(r'/home/gabriel/Documents/PhD/netket/Examples/Continuous/CS/CS_10_1d.mpack', 'rb') as file:
    vs.variables = flax.serialization.from_bytes(vs.variables, file.read())
"""
op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

gs = nk.VMC(ha, op, sab, variational_state=vs)  # , preconditioner=sr)
gs.run(n_iter=100, callback=mycb, out="test")
