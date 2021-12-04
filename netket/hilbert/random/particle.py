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
import jax
from jax import numpy as jnp

from netket.hilbert import ContinuousHilbert, Particle
from netket.utils.dispatch import dispatch


@dispatch
def random_state(hilb: Particle, key, batches: int, *, dtype):
    """Positions particles w.r.t. normal distribution,
    if no periodic boundary conditions are applied
    in a spatial dimension. Otherwise the particles are
    positioned evenly along the box from 0 to L, with Gaussian noise
    of certain width."""
    pbc = jnp.array(hilb.n_particles * hilb.pbc)
    boundary = jnp.tile(pbc, (batches, 1))

    Ls = jnp.array(hilb.n_particles * hilb.extent)
    modulus = jnp.where(jnp.equal(pbc, False), jnp.inf, Ls)

    gaussian = jax.random.normal(key, shape=(batches, hilb.size))
    width = jnp.min(modulus) / (4.0 * hilb.n_particles)
    # The width gives the noise level. In the periodic case the
    # particles are evenly distributed between 0 and min(L). The
    # distance between the particles coordinates is therefore given by
    # min(L) / hilb.N. To avoid particles to have coincident
    # positions the noise level should be smaller than half this distance.
    # We choose width = min(L) / (4*hilb.N)
    noise = gaussian * width
    uniform = jnp.tile(jnp.linspace(0.0, jnp.min(modulus), hilb.size), (batches, 1))

    rs = jnp.where(jnp.equal(boundary, False), gaussian, (uniform + noise) % modulus)

    return jnp.asarray(rs, dtype=dtype)


@dispatch
def flip_state_scalar(hilb: ContinuousHilbert, key, x, i):
    raise TypeError(
        "Flipping state is undefined for continuous Hilbert spaces. "
        "(Maybe you tried using `MetropolisLocal` on a continuous Hilbert space? "
        "This won't work because 'flipping' a continuous variable is not defined. "
        "You should try a different sampler.)"
    )
