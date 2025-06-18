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

import numpy as np

from netket import jax as nkjax
from netket.experimental.hilbert import ContinuousHilbert, Particle, ParticleSet
from netket.utils.dispatch import dispatch


def take_sub(key, x, n):
    key, subkey = jax.random.split(key)
    ind = jax.random.choice(
        subkey, jnp.arange(0, x.shape[0], 1), replace=False, shape=(n,)
    )
    return x[ind, :]


@dispatch
def random_state(hilb: Particle, key, batches: int, *, dtype):
    """If no periodic boundary conditions are present particles are positioned normally distributed around the origin.

    If periodic boundary conditions are present the particles are positioned uniformly inside the box and a small
    gaussian noise is added on top.
    If periodic boundary conditions are chosen only for certain dimensions, the periodic initialization is used for
    all of those dimensions and the free initialization is used for all the other ones.
    """
    sdim = hilb.spatial_dimension
    n_particles = hilb.n_particles

    pbc_vec = np.array(list(hilb.geometry.pbc) * n_particles)
    boundary = np.tile(pbc_vec, (batches, 1))

    # TODO generalize to particles with different domains
    Ls = np.array(list(hilb.domain) * n_particles)

    # If we have PBCs this defines the size of the Gaussian noise in the particle positions.
    # Without PBCs this is never needed (initialization with random Gaussian)
    # and we can set a random number (here 1.)
    modulus = np.where(np.equal(pbc_vec, False), 1.0, Ls)

    min_modulus = np.min(modulus)

    # use real dtypes because this does not work with complex ones.
    gaussian = jax.random.normal(
        key, shape=(batches, hilb.size), dtype=nkjax.dtype_real(dtype)
    )

    width = min_modulus / (4.0 * n_particles)
    # The width gives the noise level. In the periodic case the
    # particles are evenly distributed between 0 and min(L). The
    # distance between the particles coordinates is therefore given by
    # min(L) / hilb.N. To avoid particles to have coincident
    # positions the noise level should be smaller than half this distance.
    # We choose width = min(L) / (4*hilb.N)
    noise = gaussian * width

    key = jax.random.split(key, num=batches)
    n = int(np.ceil(n_particles ** (1 / sdim)))
    xs = jnp.linspace(0, min(hilb.domain), n)
    uniform = jnp.array(jnp.meshgrid(*(sdim * [xs]))).reshape(-1, sdim)
    uniform = jnp.tile(uniform, (batches, 1, 1))

    uniform = jax.vmap(take_sub, in_axes=(0, 0, None))(
        key, uniform, n_particles
    ).reshape(batches, -1)
    rs = jnp.where(np.equal(boundary, False), gaussian, (uniform + noise) % modulus)

    return jnp.asarray(rs, dtype=dtype)


@dispatch
def random_state(hilb: ParticleSet, key, batches: int, *, dtype):  # noqa: F811
    return random_state(hilb._impl, key, batches, dtype=dtype)


@dispatch
def flip_state_scalar(hilb: ContinuousHilbert, key, x, i):
    raise TypeError(
        "Flipping state is undefined for continuous Hilbert spaces. "
        "(Maybe you tried using `MetropolisLocal` on a continuous Hilbert space? "
        "This won't work because 'flipping' a continuous variable is not defined. "
        "You should try a different sampler.)"
    )
