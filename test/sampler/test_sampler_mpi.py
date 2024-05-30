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

import flax
from jax import numpy as jnp

from .. import common

import numpy as np
import pytest
from scipy.stats import combine_pvalues, chisquare, multivariate_normal, kstest
import jax
from jax.nn.initializers import normal

import netket as nk
from netket.hilbert import DiscreteHilbert, Particle
from netket.utils import array_in, mpi
from netket.jax.sharding import device_count_per_rank

from netket import experimental as nkx


pytestmark = common.onlyif_mpi

nk.config.update("NETKET_EXPERIMENTAL", True)
np.random.seed(1234)

WEIGHT_SEED = 1234
SAMPLER_SEED = 15324


def test_metropolis_numpy_works():
    hi = nk.hilbert.Spin(s=0.5, N=4)

    ma = nk.models.RBM()
    pars = ma.init(jax.random.key(WEIGHT_SEED), jnp.ones((2, hi.size)))

    sampler = nk.sampler.MetropolisLocalNumpy(hi, n_chains_per_rank=8)
    assert sampler.n_chains_per_rank == 8
    assert sampler.n_chains == 8 * mpi.n_nodes

    # check it works
    CHAIN_LEN = 4
    sampler_state = sampler.init_state(ma, pars, seed=SAMPLER_SEED)
    assert sampler_state.σ.shape == (sampler.n_chains_per_rank, hi.size)
    sampler_state = sampler.reset(ma, pars, state=sampler_state)
    assert sampler_state.σ.shape == (sampler.n_chains_per_rank, hi.size)

    samples, sampler_state = sampler.sample(
        ma, pars, state=sampler_state, chain_length=CHAIN_LEN
    )

    assert samples.shape == (sampler.n_chains_per_rank, CHAIN_LEN, hi.size)
    assert (
        sampler_state.n_steps_proc
        == sampler.n_chains_per_rank * CHAIN_LEN * sampler.sweep_size
    )
    assert sampler_state.n_steps == sampler_state.n_steps_proc * mpi.n_nodes
    assert sampler_state.n_accepted_proc < sampler_state.n_steps_proc
