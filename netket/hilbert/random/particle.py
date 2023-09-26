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
    shape = (batches, hilb.n_particles, hilb.geometry.dim)
    init = hilb.geometry.random_init(shape)

    rs = hilb.geometry.add(init.reshape(*shape), gaussian.reshape(*shape)).reshape(
        batches, -1
    )

    return jnp.asarray(rs, dtype=dtype)


@dispatch
def flip_state_scalar(hilb: ContinuousHilbert, key, x, i):
    raise TypeError(
        "Flipping state is undefined for continuous Hilbert spaces. "
        "(Maybe you tried using `MetropolisLocal` on a continuous Hilbert space? "
        "This won't work because 'flipping' a continuous variable is not defined. "
        "You should try a different sampler.)"
    )
