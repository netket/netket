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
import jax.numpy as jnp
import numpy as np

from flax import struct

from ..metropolis import MetropolisRule


@struct.dataclass
class GaussianRule(MetropolisRule):
    r"""
    A transition rule acting on all particle positions at once.

    New proposals of particle positions are generated according to a
    Gaussian distribution of width sigma.
    """
    sigma: float = 1.0

    def transition(rule, sampler, machine, parameters, state, key, r):
        if jnp.issubdtype(r.dtype, jnp.complexfloating):
            raise TypeError(
                "Gaussian Rule does not work with complex " "basis elements."
            )

        n_chains = r.shape[0]
        hilb = sampler.hilbert

        pbc = np.array(hilb.n_particles * hilb.pbc)
        boundary = np.tile(pbc, (n_chains, 1))

        Ls = np.array(hilb.n_particles * hilb.extent)
        modulus = np.where(np.equal(pbc, False), jnp.inf, Ls)

        prop = (
            jax.random.normal(key, shape=(n_chains, hilb.size), dtype=r.dtype)
            * rule.sigma
        )

        opt_1 = np.equal(boundary, False)
        opt_2 = np.logical_not(opt_1)
        rp = opt_1 * (r + prop) + opt_2 * ((r + prop) % modulus)

        return rp, None

    def __repr__(self):
        return "GaussianRule(sigma={})".format(self.sigma)
