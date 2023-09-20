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

from .base import MetropolisRule


@struct.dataclass
class GaussianRule(MetropolisRule):
    r"""
    A transition rule acting on all particle positions at once.

    New proposals of particle positions are generated according to a
    Gaussian distribution of width sigma.
    """

    sigma: float = 1.0
    """
    The variance of the gaussian distribution centered around the current
    configuration, used to propose new configurations.
    """

    def transition(rule, sampler, machine, parameters, state, key, r):
        if jnp.issubdtype(r.dtype, jnp.complexfloating):
            raise TypeError(
                "Gaussian Rule does not work with complex " "basis elements."
            )

        n_chains = r.shape[0]
        hilb = sampler.hilbert
        dim = hilb.geometry.dim
        prop = jax.random.normal(
            key, shape=(n_chains, hilb.size), dtype=r.dtype
        ) * jnp.asarray(rule.sigma, dtype=r.dtype)
        if "Free" in repr(hilb.geometry):
            rp = r + prop
            return rp, None

        elif "PeriodicCell" in repr(hilb.geometry):
            # transfrom to fractional coordinates
            rtemp = hilb.geometry.from_standard_to_lat(
                r.reshape(n_chains, hilb.n_particles, dim)
            ).reshape(n_chains, -1)
            rp = (rtemp + prop) % 1.0
            # backtransform to physical space
            rp = hilb.geometry.from_lat_to_standard(
                rp.reshape(n_chains, hilb.n_particles, dim)
            ).reshape(n_chains, -1)
            return rp, None

        else:
            raise NotImplementedError

    def __repr__(self):
        return f"GaussianRule(sigma={self.sigma})"
