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

import netket as nk
from netket.hilbert import DiscreteHilbert, Particle

from netket import experimental as nkx

import numpy as np
import pytest
from scipy.stats import combine_pvalues, chisquare, multivariate_normal, kstest
import jax
from jax.nn.initializers import normal

from jax.config import config

config.update("jax_enable_x64", True)


pytestmark = common.skipif_mpi

nk.config.update("NETKET_EXPERIMENTAL", True)
np.random.seed(1234)

WEIGHT_SEED = 1234
SAMPLER_SEED = 15324


samplers = {}


# TESTS FOR SPIN HILBERT
# Constructing a 1d lattice
g = nk.graph.Hypercube(length=4, n_dim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
move_op = sum([nk.operator.spin.sigmax(hi, i) for i in range(hi.size)])

hi_spin1 = nk.hilbert.Spin(s=1, N=g.n_nodes)
hib = nk.hilbert.Fock(n_max=1, N=g.n_nodes, n_particles=1)
hib_u = nk.hilbert.Fock(n_max=3, N=g.n_nodes)

samplers["Exact: Spin"] = nk.sampler.ExactSampler(hi)
samplers["Exact: Fock"] = nk.sampler.ExactSampler(hib_u)

samplers["Metropolis(Local): Spin"] = nk.sampler.MetropolisLocal(hi)

samplers["MetropolisNumpy(Local): Spin"] = nk.sampler.MetropolisLocalNumpy(hi)
# samplers["MetropolisNumpy(Local): Fock"] = nk.sampler.MetropolisLocalNumpy(hib_u)
# samplers["MetropolisNumpy(Local): Doubled-Spin"] = nk.sampler.MetropolisLocalNumpy(
#    nk.hilbert.DoubledHilbert(nk.hilbert.Spin(s=0.5, N=2))
# )

samplers["MetropolisPT(Local): Spin"] = nkx.sampler.MetropolisLocalPt(hi, n_replicas=4)
samplers["MetropolisPT(Local): Fock"] = nkx.sampler.MetropolisLocalPt(
    hib_u, n_replicas=4
)

samplers["Metropolis(Exchange): Fock-1particle"] = nk.sampler.MetropolisExchange(
    hib, graph=g
)

samplers["Metropolis(Hamiltonian,Jax): Spin"] = nk.sampler.MetropolisHamiltonian(
    hi,
    hamiltonian=ha,
    reset_chains=True,
)

samplers["Metropolis(Hamiltonian,Numpy): Spin"] = nk.sampler.MetropolisHamiltonianNumpy(
    hi,
    hamiltonian=ha,
    reset_chains=True,
)

samplers["Metropolis(Custom: Sx): Spin"] = nk.sampler.MetropolisCustom(
    hi, move_operators=move_op
)

# samplers["MetropolisPT(Custom: Sx): Spin"] = nkx.sampler.MetropolisCustomPt(hi, move_operators=move_op, n_replicas=4)

samplers["Autoregressive: Spin 1/2"] = nk.sampler.ARDirectSampler(hi)
samplers["Autoregressive: Spin 1"] = nk.sampler.ARDirectSampler(hi_spin1)
samplers["Autoregressive: Fock"] = nk.sampler.ARDirectSampler(hib_u)


# Hilbert space and sampler for particles
hi_particles = nk.hilbert.Particle(N=3, L=(np.inf,), pbc=(False,))
samplers["Metropolis(Gaussian): Gaussian"] = nk.sampler.MetropolisGaussian(
    hi_particles, sigma=1.0
)


# The following fixture initialisees a model and it's weights
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
                dtype=complex,
                kernel_init=normal(stddev=0.1),
                hidden_bias_init=normal(stddev=0.1),
            )

        # init network
        w = ma.init(jax.random.PRNGKey(WEIGHT_SEED), jnp.zeros((1, hilb.size)))

        return ma, w

    # Do something with the data
    return build_model


# The following fixture returns one sampler at a time (and iterates through)
# all samplers.
#  it skips tests according to the --sampler cmdline argument introduced in
# conftest.py
@pytest.fixture(
    params=[pytest.param(sampl, id=name) for name, sampl in samplers.items()]
)
def sampler(request):
    cmdline_sampler = request.config.getoption("--sampler").lower()
    if cmdline_sampler == "":
        return request.param
    elif cmdline_sampler in request.node.name.lower():
        return request.param
    else:
        pytest.skip("skipped from command-line argument")


@pytest.fixture(params=[pytest.param(val, id=f", mpow={val}") for val in [1, 2]])
def set_pdf_power(request):
    def fun(sampler):
        cmdline_mpow = request.config.getoption("--mpow").lower()
        if cmdline_mpow == "all":
            # Nothing to skip
            pass
        elif cmdline_mpow == "single":
            # same sampler leads to same rng
            rng = np.random.default_rng(common.hash_for_seed(sampler))
            exponent = rng.integers(1, 3)  # 1 or 2
            if exponent != request.param:
                pytest.skip(
                    "Running only 1 pdf exponent per sampler. Use --mpow=all to run all pdf exponents."
                )
        elif int(cmdline_mpow) != request.param:
            pytest.skip(f"Running only --mpow={cmdline_mpow}.")

        if isinstance(sampler, nk.sampler.ARDirectSampler) and request.param != 2:
            pytest.skip("ARDirectSampler only supports machine_pow = 2.")

        return sampler.replace(machine_pow=request.param)

    return fun


def test_states_in_hilbert(sampler, model_and_weights):
    hi = sampler.hilbert
    if isinstance(hi, DiscreteHilbert):
        all_states = hi.all_states()

        ma, w = model_and_weights(hi, sampler)

        for sample in sampler.samples(ma, w, chain_length=50):
            assert sample.shape == (sampler.n_chains, hi.size)
            for v in sample:
                assert v in all_states

    elif isinstance(hi, Particle):
        ma, w = model_and_weights(hi, sampler)

        for sample in sampler.samples(ma, w, chain_length=50):
            assert sample.shape == (sampler.n_chains, hi.size)

    # if hasattr(sa, "acceptance"):
    #    assert np.min(sampler.acceptance) >= 0 and np.max(sampler.acceptance) <= 1.0


def findrng(rng):
    if hasattr(rng, "_bit_generator"):
        return rng._bit_generator.state["state"]
    else:
        return rng


# Mark tests that we know are failing on correctedness
def failing_test(sampler):
    return False


@pytest.fixture(
    params=[
        pytest.param(
            sampl,
            id=name,
            marks=pytest.mark.xfail(reason="MUSTFIX: this sampler is known to fail")
            if failing_test(sampl)
            else [],
        )
        for name, sampl in samplers.items()
    ]
)
def sampler_c(request):
    cmdline_sampler = request.config.getoption("--sampler").lower()
    if cmdline_sampler == "":
        return request.param
    elif cmdline_sampler in request.node.name.lower():
        return request.param
    else:
        pytest.skip("skipped from command-line argument")


# Testing that samples generated from direct sampling are compatible with those
# generated by markov chain sampling
# here we use a combination of power divergence tests
def test_correct_sampling(sampler_c, model_and_weights, set_pdf_power):
    sampler = set_pdf_power(sampler_c)

    hi = sampler.hilbert
    if isinstance(hi, DiscreteHilbert):
        n_states = hi.n_states

        ma, w = model_and_weights(hi, sampler)

        n_samples = max(40 * n_states, 100)

        ps = (
            np.absolute(nk.nn.to_array(hi, ma, w, normalize=False))
            ** sampler.machine_pow
        )
        ps /= ps.sum()

        n_rep = 6
        pvalues = np.zeros(n_rep)

        sampler_state = sampler.init_state(ma, w, seed=SAMPLER_SEED)

        for jrep in range(n_rep):
            sampler_state = sampler.reset(ma, w, state=sampler_state)

            # Burnout phase
            samples, sampler_state = sampler.sample(
                ma, w, state=sampler_state, chain_length=n_samples // 100
            )

            assert samples.shape == (
                n_samples // 100,
                sampler.n_chains,
                hi.size,
            )
            samples, sampler_state = sampler.sample(
                ma, w, state=sampler_state, chain_length=n_samples
            )

            assert samples.shape == (n_samples, sampler.n_chains, hi.size)

            sttn = hi.states_to_numbers(np.asarray(samples.reshape(-1, hi.size)))
            n_s = sttn.size

            # fill in the histogram for sampler
            unique, counts = np.unique(sttn, return_counts=True)
            hist_samp = np.zeros(n_states)
            hist_samp[unique] = counts

            # expected frequencies
            f_exp = n_s * ps
            statistics, pvalues[jrep] = chisquare(hist_samp, f_exp=f_exp)

        s, pval = combine_pvalues(pvalues, method="fisher")
        assert pval > 0.01 or np.max(pvalues) > 0.01

    elif isinstance(hi, Particle):
        ma, w = model_and_weights(hi, sampler)
        n_samples = 5000
        n_discard = 2000
        n_rep = 6
        pvalues = np.zeros(n_rep)

        sampler_state = sampler.init_state(ma, w, seed=SAMPLER_SEED)
        for jrep in range(n_rep):
            sampler_state = sampler.reset(ma, w, state=sampler_state)

            # Burnout phase
            samples, sampler_state = sampler.sample(
                ma, w, state=sampler_state, chain_length=n_discard
            )

            assert samples.shape == (
                n_discard,
                sampler.n_chains,
                hi.size,
            )
            samples, sampler_state = sampler.sample(
                ma, w, state=sampler_state, chain_length=n_samples
            )

            assert samples.shape == (n_samples, sampler.n_chains, hi.size)

            samples = samples.reshape(-1, samples.shape[-1])

            dist = multivariate_normal(
                mean=np.zeros(samples.shape[-1]),
                cov=np.linalg.inv(
                    sampler.machine_pow
                    * np.dot(w["params"]["kernel"].T, w["params"]["kernel"])
                ),
            )
            exact_samples = dist.rvs(size=samples.shape[0])

            counts, bins = np.histogramdd(samples, bins=10)
            counts_exact, _ = np.histogramdd(exact_samples, bins=bins)

            statistics, pvalues[jrep] = kstest(
                counts.reshape(-1), counts_exact.reshape(-1)
            )

        s, pval = combine_pvalues(pvalues, method="fisher")
        assert pval > 0.01 or np.max(pvalues) > 0.01


def test_throwing(model_and_weights):
    with pytest.raises(TypeError):
        nk.sampler.MetropolisHamiltonian(
            hi,
            hamiltonian=10,
            reset_chains=True,
        )

    with pytest.raises(ValueError):
        sampler = nk.sampler.MetropolisHamiltonian(
            nk.hilbert.DoubledHilbert(hi),
            hamiltonian=ha,
            reset_chains=True,
        )

        ma, w = model_and_weights(hi)

        # test raising of init state
        sampler.init_state(ma, w, seed=SAMPLER_SEED)

    with pytest.raises(ValueError):
        sampler = nk.sampler.MetropolisHamiltonianNumpy(
            nk.hilbert.Fock(3) ** hi.size,
            hamiltonian=ha,
            reset_chains=True,
        )

        ma, w = model_and_weights(hi)

        # test raising of init state
        sampler.init_state(ma, w, seed=SAMPLER_SEED)

    with pytest.raises(flax.errors.ScopeParamShapeError):
        sampler = nk.sampler.MetropolisHamiltonianNumpy(
            nk.hilbert.DoubledHilbert(hi),
            hamiltonian=ha,
            reset_chains=True,
        )

        ma, w = model_and_weights(hi)

        # test raising of init state
        sampler.init_state(ma, w, seed=SAMPLER_SEED)

    with pytest.raises(ValueError):
        nk.sampler.ARDirectSampler(hi, machine_pow=1)

    # MetropolisLocal should not work with continuous Hilbert spaces
    with pytest.raises(TypeError):
        sampler = nk.sampler.MetropolisLocal(hi_particles)

        ma, w = model_and_weights(hi)

        sampler.sample(ma, w, seed=SAMPLER_SEED)


def test_exact_sampler(sampler):
    known_exact_samplers = (nk.sampler.ExactSampler, nk.sampler.ARDirectSampler)
    if isinstance(sampler, known_exact_samplers):
        assert sampler.is_exact is True
        assert sampler.n_chains_per_rank == 1
    else:
        assert sampler.is_exact is False
        assert sampler.n_chains_per_rank == 16
