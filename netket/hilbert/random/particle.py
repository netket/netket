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
from netket.hilbert import ContinuousHilbert, Particle
from netket.utils.dispatch import dispatch


def take_sub(key, x, n):
    key, subkey = jax.random.split(key)
    ind = jax.random.choice(
        subkey, jnp.arange(0, x.shape[0], 1), replace=False, shape=(n,)
    )
    return x[ind, :]


@dispatch
def random_state(hilb: Particle, key, batches: int, *, dtype):
    """The geometry object in hilbert determines which random_state is used. If there is a periodic box, the random
    state samples fractional coordinates by positioning the particles uniformly inside a box of size 1
    and adding a small gaussian noise on top. These numbers are then mapped to physical space using the lattice.
    If no periodic boundary conditions are present particles are positioned normally distributed around the origin.
    """
    # use real dtypes because this does not work with complex ones.
    gaussian = jax.random.normal(
        key, shape=(batches, hilb.size), dtype=nkjax.dtype_real(dtype)
    )
    dim = hilb.geometry.dim
    uniform = jnp.zeros_like(gaussian)
    if all(map(lambda x: x is True, hilb.geometry.pbc)):
        width = jnp.linalg.norm(hilb.geometry.lattice[0]) / (4.0 * hilb.n_particles)
        # The width gives the noise level. In the periodic case the
        # particles are evenly distributed between 0 and the boundary of the box. The
        # distance between the particles coordinates is therefore given by
        # size of box / hilb.N. To avoid particles to have coincident
        # positions the noise level should be smaller than half this distance.
        # We choose width = min(L) / (4*hilb.N)
        gaussian = gaussian * width
        # make uniform grid
        key = jax.random.split(key, num=batches)
        dim = hilb.geometry.dim
        n = int(np.ceil(hilb.n_particles ** (1 / dim)))
        xs = jnp.linspace(0, 1, n)
        uniform = jnp.array(jnp.meshgrid(*(dim * [xs]))).T.reshape(-1, dim)
        uniform = jnp.tile(uniform, (batches, 1, 1))
        uniform = jax.vmap(take_sub, in_axes=(0, 0, None))(
            key, uniform, hilb.n_particles
        ).reshape(batches, -1)

    # fold positions to the given periodic lattice using standard to lattice mapping
    rs = hilb.geometry.from_lat_to_standard(
        (uniform + gaussian).reshape(batches, hilb.n_particles, dim)
    ).reshape(batches, -1)
    return jnp.asarray(rs, dtype=dtype)


@dispatch
def flip_state_scalar(hilb: ContinuousHilbert, key, x, i):
    raise TypeError(
        "Flipping state is undefined for continuous Hilbert spaces. "
        "(Maybe you tried using `MetropolisLocal` on a continuous Hilbert space? "
        "This won't work because 'flipping' a continuous variable is not defined. "
        "You should try a different sampler.)"
    )
