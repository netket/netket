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
import jax
from jax.nn.initializers import normal

import netket as nk
from netket.hilbert import Particle


np.random.seed(1234)

WEIGHT_SEED = 1234
SAMPLER_SEED = 15324


# Initializations raising errors
@common.skipif_mpi
def test_wrong_initialization():
    g = nk.graph.Hypercube(length=4, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    hib_u = nk.hilbert.Fock(n_max=3, N=g.n_nodes)

    for n_replicas in [-3, 0, 5, 2.1]:
        with pytest.raises(
            ValueError,
        ):
            sa = nk.sampler.ParallelTemperingLocal(
                hi,
                n_replicas=n_replicas,
                sweep_size=hib_u.size * 4,
            )

    with pytest.raises(ValueError):
        sa = nk.sampler.ParallelTemperingLocal(
            hi,
            betas="custom",
            sweep_size=hib_u.size * 4,
        )

    for betas in [[1.1, 0.5], [-1.0, 0.2], [0.0, 1.0]]:
        with pytest.raises(ValueError):
            sa = nk.sampler.ParallelTemperingLocal(
                hi,
                betas=betas,
                sweep_size=hib_u.size * 4,
            )

    sa = nk.sampler.ParallelTemperingLocal(
        hi,
        sweep_size=hib_u.size * 4,
    )
    assert (sa.sorted_betas == 1 - np.arange(32) / 32).all()


# Verify the possibility to initialize in multiple ways
@pytest.mark.parametrize("n_replicas", [None, 32])
@pytest.mark.parametrize("betas", ["linear", "logarithmic"])
def test_initialization_beta_distribution(model_and_weights, n_replicas, betas):
    g = nk.graph.Hypercube(length=4, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    hib_u = nk.hilbert.Fock(n_max=3, N=g.n_nodes)

    sa = nk.sampler.ParallelTemperingLocal(
        hi,
        n_replicas=n_replicas,
        betas=betas,
        sweep_size=hib_u.size * 4,
    )
    assert sa.n_replicas == 32
    assert sa.sorted_betas.shape == (32,)
    assert sa.sorted_betas[0] == 1 and sa.sorted_betas[-1] > 0

    ma, w = model_and_weights(hi, sa)

    sampler_state = sa.init_state(ma, w, seed=SAMPLER_SEED)
    assert sampler_state.beta.shape == (sa.n_batches // sa.n_replicas, sa.n_replicas)


betas_list = [
    pytest.param(1 - (np.arange(32) / 32) ** 3, id="cubic"),
    pytest.param(np.linspace(0.1, 1, 32)[::-1], id="shuffled"),
]


# Verify the possibility to initialize in multiple ways
@pytest.mark.parametrize("betas", betas_list)
def test_initialization_beta_list(model_and_weights, betas):
    g = nk.graph.Hypercube(length=4, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    hib_u = nk.hilbert.Fock(n_max=3, N=g.n_nodes)

    sa = nk.sampler.ParallelTemperingLocal(
        hi,
        betas=betas,
        sweep_size=hib_u.size * 4,
    )
    assert sa.n_replicas == 32
    assert sa.sorted_betas.shape == (32,)
    assert sa.sorted_betas[0] == 1 and sa.sorted_betas[-1] > 0

    ma, w = model_and_weights(hi, sa)

    sampler_state = sa.init_state(ma, w, seed=SAMPLER_SEED)
    assert sampler_state.beta.shape == (sa.n_batches // sa.n_replicas, sa.n_replicas)


# This test verifies that the acceptance is indeed a float
# It also verifies that its value is 1 for a state with flat distribution
def test_acceptance():
    g = nk.graph.Hypercube(length=4, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    ma = nk.models.Jastrow(
        kernel_init=flax.linen.initializers.constant(0.0)
    )  # |psi> = |+>

    sa = nk.sampler.ParallelTemperingLocal(
        hi,
        n_replicas=4,
        sweep_size=hi.size * 4,
    )
    vs = nk.vqs.MCState(sa, ma, n_samples=1000, seed=WEIGHT_SEED)
    vs.sample()

    assert jnp.isclose(vs.sampler_state.acceptance, 1.0)


# The following fixture initialises a model and it's weights
# for tests that require it.
@pytest.fixture
def model_and_weights(request):
    def build_model(hilb, sampler=None):
        if isinstance(sampler, nk.sampler.ARDirectSampler):
            ma = nk.models.ARNNDense(
                hilbert=hilb, machine_pow=sampler.machine_pow, layers=3, features=5
            )
        elif isinstance(hilb, Particle):
            ma = nk.models.Gaussian()
        else:
            # Build RBM by default
            ma = nk.models.RBM(
                alpha=1,
                param_dtype=complex,
                kernel_init=normal(stddev=0.1),
                hidden_bias_init=normal(stddev=0.1),
            )

        # init network
        w = ma.init(jax.random.PRNGKey(WEIGHT_SEED), jnp.zeros((1, hilb.size)))

        return ma, w

    # Do something with the data
    return build_model


@common.onlyif_mpi
def test_multiplerules_pt_mpi(model_and_weights):
    g = nk.graph.Hypercube(length=4, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
    hib_u = nk.hilbert.Fock(n_max=3, N=g.n_nodes)

    sa = nk.sampler.ParallelTemperingSampler(
        hi,
        rule=nk.sampler.rules.MultipleRules(
            [nk.sampler.rules.LocalRule(), nk.sampler.rules.HamiltonianRule(ha)],
            [0.8, 0.2],
        ),
        n_replicas=4,
        sweep_size=hib_u.size * 4,
    )

    ma, w = model_and_weights(hi, sa)

    sampler_state = sa.init_state(ma, w, seed=SAMPLER_SEED)
    sampler_state = sa.reset(ma, w, state=sampler_state)
    samples, sampler_state = sa.sample(
        ma,
        w,
        state=sampler_state,
        chain_length=10,
    )
    assert samples.shape == (sa.n_batches // sa.n_replicas, 10, hi.size)
