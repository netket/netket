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
from netket import experimental as nkx

import jax.numpy as jnp


def v(x):
    return jnp.linalg.norm(x) ** 2


space = nkx.geometry.Cell(d=3, L=(jnp.inf, jnp.inf, jnp.inf), pbc=False)
hilb = nkx.hilbert.Particle(N=10, geometry=space)

sab = nk.sampler.MetropolisGaussian(hilb, sigma=0.1, n_chains=16, sweep_size=32)

ekin = nk.operator.KineticEnergy(hilb, mass=1.0)
pot = nk.operator.PotentialEnergy(hilb, v)
ha = ekin + 0.5 * pot

model = nk.models.Gaussian(param_dtype=float)

vs = nk.vqs.MCState(sab, model, n_samples=10**4, n_discard_per_chain=2000)

op = nk.optimizer.Sgd(0.05)

gs = nk.driver.VMC_SR(ha, op, variational_state=vs, diag_shift=0.01)
gs.run(n_iter=100, out="HO_10_3d")
