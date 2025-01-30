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

import pytest

import numpy as np
from scipy.stats import combine_pvalues, chisquare, multivariate_normal, kstest

import jax
from jax.nn.initializers import normal

import netket as nk
from netket import config
from netket.hilbert import DiscreteHilbert, Particle
from netket.utils import array_in, mpi
from netket.jax.sharding import device_count_per_rank


from .. import common

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
hi_fermion = nk.hilbert.SpinOrbitalFermions(g.n_nodes, n_fermions=2)
hi_fermion_spin = nk.hilbert.SpinOrbitalFermions(
    g.n_nodes, s=1 / 2, n_fermions_per_spin=(2, 2)
)
hi_fermion_spin_higher = nk.hilbert.SpinOrbitalFermions(
    g.n_nodes, s=3 / 2, n_fermions_per_spin=(2, 2, 1, 1)
)

samplers["Exact: Spin"] = nk.sampler.ExactSampler(hi)
samplers["Exact: Fock"] = nk.sampler.ExactSampler(hib_u)

samplers["Metropolis(Local): Spin"] = nk.sampler.MetropolisLocal(hi)
samplers["Metropolis(Local): Spin-chunked"] = nk.sampler.MetropolisLocal(
    hi, chunk_size=8
)

samplers["MetropolisNumpy(Local): Spin"] = nk.sampler.MetropolisLocalNumpy(hi)
samplers["MetropolisNumpy(Local): Spin-chunked"] = nk.sampler.MetropolisLocalNumpy(
    hi, chunk_size=8
)
# samplers["MetropolisNumpy(Local): Fock"] = nk.sampler.MetropolisLocalNumpy(hib_u)
# samplers["MetropolisNumpy(Local): Doubled-Spin"] = nk.sampler.MetropolisLocalNumpy(
#    nk.hilbert.DoubledHilbert(nk.hilbert.Spin(s=0.5, N=2))
# )

samplers["MetropolisPT(Local): Spin"] = nk.sampler.ParallelTemperingLocal(
    hi, n_replicas=4, sweep_size=hi.size * 4
)
samplers["MetropolisPT(Local): Fock"] = nk.sampler.ParallelTemperingLocal(
    hib_u, n_replicas=4, sweep_size=hib_u.size * 4
)

samplers["Metropolis(Exchange): Fock-1particle"] = nk.sampler.MetropolisExchange(
    hib, graph=g
)

if not config.netket_experimental_sharding:
    samplers["Metropolis(Hamiltonian,numba operator): Spin"] = (
        nk.sampler.MetropolisHamiltonian(
            hi,
            hamiltonian=ha,
            reset_chains=True,
        )
    )

samplers["Metropolis(ParticleExchange): SpinOrbitalFermions"] = (
    nk.sampler.MetropolisFermionHop(hi_fermion, graph=g)
)
samplers["Metropolis(ParticleExchange,Spinful): SpinOrbitalFermions"] = (
    nk.sampler.MetropolisFermionHop(hi_fermion_spin, graph=g, spin_symmetric=True)
)
samplers["Metropolis(ParticleExchange,Spinful=3/2): SpinOrbitalFermions"] = (
    nk.sampler.MetropolisFermionHop(
        hi_fermion_spin_higher, graph=g, spin_symmetric=True
    )
)

samplers["Metropolis(Hamiltonian,Numpy): Spin"] = nk.sampler.MetropolisHamiltonianNumpy(
    hi,
    hamiltonian=ha,
    reset_chains=True,
)

ha_jax = nk.operator.IsingJax(hilbert=hi, graph=g, h=1.0)

samplers["Metropolis(Hamiltonian, jax operator): Spin"] = (
    nk.sampler.MetropolisHamiltonian(
        hi,
        hamiltonian=ha_jax,
        reset_chains=True,
    )
)

samplers["Metropolis(Custom: Sx): Spin"] = nk.sampler.MetropolisCustom(
    hi, move_operators=move_op
)

# MultipleRules sampler
samplers["Metropolis(MultipleRules[Local,Local]): Spin"] = nk.sampler.MetropolisSampler(
    hi,
    nk.sampler.rules.MultipleRules(
        [nk.sampler.rules.LocalRule(), nk.sampler.rules.LocalRule()], [0.8, 0.2]
    ),
)
if not config.netket_experimental_sharding:
    samplers["Metropolis(MultipleRules[Local,Hamiltonian]): Spin"] = (
        nk.sampler.MetropolisSampler(
            hi,
            nk.sampler.rules.MultipleRules(
                [nk.sampler.rules.LocalRule(), nk.sampler.rules.HamiltonianRule(ha)],
                [0.8, 0.2],
            ),
        )
    )


samplers["Autoregressive: Spin 1/2"] = nk.sampler.ARDirectSampler(hi)
samplers["Autoregressive: Spin 1"] = nk.sampler.ARDirectSampler(hi_spin1)
samplers["Autoregressive: Fock"] = nk.sampler.ARDirectSampler(hib_u)


# Hilbert space and sampler for particles
hi_particles = nk.hilbert.Particle(N=3, L=jnp.inf, pbc=False)
samplers["Metropolis(Gaussian): Gaussian"] = nk.sampler.MetropolisGaussian(
    hi_particles, sigma=1.0, sweep_size=hi_particles.size * 10
)
samplers["Metropolis(AdjustedLangevin): AdjustedLangevin"] = (
    nk.sampler.MetropolisAdjustedLangevin(
        hi_particles, dt=0.1, sweep_size=hi_particles.size
    )
)
samplers["Metropolis(AdjustedLangevin): AdjustedLangevin chunk_size"] = (
    nk.sampler.MetropolisAdjustedLangevin(hi_particles, dt=0.1, chunk_size=16)
)

# TensorHilbert sampler
hi = nk.hilbert.Spin(0.5, 4) * nk.hilbert.Fock(3)
samplers["Metropolis(TensorRule): Spin x Fock"] = nk.sampler.MetropolisSampler(
    hi,
    nk.sampler.rules.TensorRule(
        hi, [nk.sampler.rules.LocalRule(), nk.sampler.rules.LocalRule()]
    ),
)

# TensorHilbert sampler
hi = nk.hilbert.Spin(0.5, 4) * nk.hilbert.Fock(3)
ha = sum(nk.operator.spin.sigmax(nk.hilbert.Spin(0.5, 4), i) for i in range(4))
if not config.netket_experimental_sharding:
    samplers["Metropolis(TensorRule): Spin x Fock"] = nk.sampler.MetropolisSampler(
        hi,
        nk.sampler.rules.TensorRule(
            hi, [nk.sampler.rules.HamiltonianRule(ha), nk.sampler.rules.LocalRule()]
        ),
    )


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


# The following fixture returns one sampler at a time (and iterates through)
# all samplers.
# it skips tests according to the --sampler cmdline argument introduced in
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


@common.skipif_distributed
def test_states_in_hilbert(sampler, model_and_weights):
    hi = sampler.hilbert
    chain_length = 50
    if isinstance(hi, DiscreteHilbert):
        all_states = hi.all_states()

        ma, w = model_and_weights(hi, sampler)

        samples, _ = sampler.sample(ma, w, chain_length=chain_length)
        assert samples.shape == (sampler.n_chains, chain_length, hi.size)
        for sample in np.asarray(samples).reshape(-1, hi.size):
            assert array_in(sample, all_states)

    elif isinstance(hi, Particle):
        ma, w = model_and_weights(hi, sampler)
        samples, _ = sampler.sample(ma, w, chain_length=chain_length)
        assert samples.shape == (sampler.n_chains, chain_length, hi.size)

    # if hasattr(sa, "acceptance"):
    #    assert np.min(sampler.acceptance) >= 0 and np.max(sampler.acceptance) <= 1.0


def findrng(rng):
    if hasattr(rng, "_bit_generator"):
        return rng._bit_generator.state["state"]
    else:
        return rng


@pytest.fixture(
    params=[
        pytest.param(
            sampl,
            id=name,
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


# !!WARN!! TODO: Flaky test
# This tests do not take into account the fact that our samplers do not necessarily
# produce samples which are uncorrelated. So unless the autocorrelation time is 0, we
# are bound to fail such tests. We should account for that.
@common.skipif_distributed
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
                sampler.n_chains,
                n_samples // 100,
                hi.size,
            )
            samples, sampler_state = sampler.sample(
                ma, w, state=sampler_state, chain_length=n_samples
            )

            assert samples.shape == (sampler.n_chains, n_samples, hi.size)

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
        # TODO: Find periodic distribution that can be exactly sampled and do the same test.

        ma, w = model_and_weights(hi, sampler)
        n_samples = 5000
        n_discard = 2 * 1024
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
                sampler.n_chains,
                n_discard,
                hi.size,
            )
            samples, sampler_state = sampler.sample(
                ma,
                w,
                state=sampler_state,
                chain_length=n_samples,
            )

            assert samples.shape == (sampler.n_chains, n_samples, hi.size)

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


@pytest.mark.skipif(
    not nk.config.netket_experimental_sharding, reason="Only run with sharding"
)
@pytest.mark.skipif(jax.device_count() < 2, reason="Only run with >1 device")
def test_sampling_sharded_not_communicating(
    sampler_c, model_and_weights, set_pdf_power
):
    if isinstance(sampler_c, nk.sampler.MetropolisNumpy):
        pytest.skip("Not jit compatible")
    if isinstance(sampler_c, nk.sampler.ExactSampler):
        pytest.xfail("Error logic communicates")

    sampler = set_pdf_power(sampler_c)
    hi = sampler.hilbert
    ma, w = model_and_weights(hi, sampler)
    sampler_state = sampler.init_state(ma, w, seed=SAMPLER_SEED)
    samples, sampler_state = sampler.sample(ma, w, state=sampler_state, chain_length=1)

    sample_jit = jax.jit(
        sampler.sample, static_argnums=0, static_argnames="chain_length"
    )
    complied = sample_jit.lower(ma, w, state=sampler_state, chain_length=1).compile()
    txt = complied.as_text()
    for o in [
        "all-reduce",
        "collective-permute",
        "all-gather",
        "all-to-all",
        "reduce-scatter",
    ]:
        for l in txt.split("\n"):
            if "equinox" in l:
                # allow equinox error_if all-gather
                continue
            assert o not in l


@common.skipif_distributed
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

    # Shouldn't accept chunk sizes that don't divide n_samples_per_rank
    with pytest.raises(ValueError):
        sampler = nk.sampler.MetropolisLocal(hi, chunk_size=5)

    with pytest.raises(ValueError):
        sampler = nk.sampler.MetropolisLocalNumpy(hi, chunk_size=5)


@common.skipif_distributed
def test_setup_throwing_tensorrule():
    # TensorHilbert sampler
    hi = nk.hilbert.Spin(0.5, 4) * nk.hilbert.Fock(3)
    ha = sum(nk.operator.spin.sigmax(nk.hilbert.Spin(0.5, 4), i) for i in range(4))

    rule1 = nk.sampler.rules.HamiltonianRule(ha)
    rule2 = nk.sampler.rules.LocalRule()

    with pytest.raises(TypeError):
        # Hilbert not TensorHilbert
        nk.sampler.rules.TensorRule(nk.hilbert.Spin(0.5, 5), [rule1, rule1, rule2])
    with pytest.raises(TypeError):
        # not list of rules
        nk.sampler.rules.TensorRule(hi, rule1)
    with pytest.raises(TypeError):
        # Not good types
        nk.sampler.rules.TensorRule(hi, [rule1, 2])
    with pytest.raises(ValueError):
        # length mismatch
        nk.sampler.rules.TensorRule(hi, [rule1, rule1, rule2])


@common.skipif_distributed
def test_setup_throwing_multiplerules():
    rule1 = nk.sampler.rules.LocalRule()
    rule2 = nk.sampler.rules.LocalRule()

    with pytest.raises(ValueError):
        # length mismatch
        nk.sampler.rules.MultipleRules([rule1, rule2], [0.5, 0.25, 0.25])
    with pytest.raises(ValueError):
        # not summing to 1
        nk.sampler.rules.MultipleRules([rule1, rule2], [0.5, 0.25])
    with pytest.raises(TypeError):
        # wrong types
        nk.sampler.rules.MultipleRules([rule1, 2], [0.5, 0.5])
    with pytest.raises(TypeError):
        # wrong types
        nk.sampler.rules.MultipleRules(rule1, [0.5, 0.5])


@common.skipif_distributed
def test_exact_sampler(sampler):
    known_exact_samplers = (nk.sampler.ExactSampler, nk.sampler.ARDirectSampler)
    if isinstance(sampler, known_exact_samplers):
        assert sampler.is_exact is True
        assert sampler.n_chains_per_rank == 1
    else:
        assert sampler.is_exact is False
        assert sampler.n_chains == 16 * mpi.n_nodes * device_count_per_rank()


@common.skipif_distributed
def test_fermions_spin_exchange():
    # test that the graph correctly creates a disjoint graph for the spinful case
    g = nk.graph.Hypercube(length=4, n_dim=1)
    hi_fermion_spin = nk.hilbert.SpinOrbitalFermions(
        g.n_nodes, s=1 / 2, n_fermions_per_spin=(2, 2)
    )

    sampler = nk.sampler.MetropolisFermionHop(
        hi_fermion_spin, graph=g, spin_symmetric=False
    )
    nodes = np.unique(sampler.rule.clusters)
    assert np.allclose(nodes, np.arange(g.n_nodes))

    sampler = nk.sampler.MetropolisFermionHop(
        hi_fermion_spin, graph=g, spin_symmetric=True
    )
    nodes = np.unique(sampler.rule.clusters)
    assert np.allclose(nodes, np.arange(hi_fermion_spin.size))

    hi_fermion_spin_higher = nk.hilbert.SpinOrbitalFermions(
        g.n_nodes, s=3 / 2, n_fermions_per_spin=(2, 2, 1, 1)
    )

    sampler = nk.sampler.MetropolisFermionHop(
        hi_fermion_spin_higher, graph=g, spin_symmetric=False
    )
    nodes = np.unique(sampler.rule.clusters)
    assert np.allclose(nodes, np.arange(g.n_nodes))

    sampler = nk.sampler.MetropolisFermionHop(
        hi_fermion_spin_higher, graph=g, spin_symmetric=True
    )
    nodes = np.unique(sampler.rule.clusters)
    assert np.allclose(nodes, np.arange(hi_fermion_spin_higher.size))


@common.skipif_distributed
def test_multiplerules_pt(model_and_weights):
    hi = ha.hilbert
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
    assert samples.shape == (sa.n_chains, 10, hi.size)


def test_hamiltonian_jax_sampler_isleaf():
    g = nk.graph.Hypercube(length=4, n_dim=1, pbc=True)

    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    rule1 = nk.sampler.rules.HamiltonianRule(
        nk.operator.IsingJax(hilbert=hi, graph=g, h=1.0)
    )
    rule2 = nk.sampler.rules.HamiltonianRule(
        nk.operator.IsingJax(hilbert=hi, graph=g, h=1.0)
    )
    leaf1, struct1 = jax.tree_util.tree_flatten(rule1)
    leaf2, struct2 = jax.tree_util.tree_flatten(rule2)

    # if the structures are identical, the operators must have been unpacked.
    assert struct1 == struct2
    assert hash(struct1) == hash(struct2)

    # check contained in leafs (this only works because the arrays are identically the same, but it
    # is enough for this check):
    for leaf in jax.tree_util.tree_leaves(rule1.operator):
        found = False
        for l in leaf1:
            if leaf is l:
                found = True
                break
        # If this fails, it is either because the operator is not a leaf ot the rule, or because jax changed
        # some internals and the flattening does not return identical arrays anymore.
        assert found


# we've got chunked samplers for these two
@pytest.mark.parametrize(
    "sampler_type", ["MetropolisNumpy(Local): Spin", "Metropolis(Local): Spin"]
)
@common.skipif_distributed
def test_chunking_invariant(model_and_weights, sampler_type):
    sa = samplers[sampler_type]

    if isinstance(sa, nk.sampler.MetropolisNumpy):
        if nk.config.netket_experimental_sharding:
            pytest.xfail(reason="TODO: to be investigated.")

    hi = sa.hilbert
    ma, w = model_and_weights(hi, sa)

    sampler_state = sa.init_state(ma, w, seed=SAMPLER_SEED)
    sampler_state = sa.reset(ma, w, state=sampler_state)
    samples, sampler_state = sa.sample(
        ma,
        w,
        state=sampler_state,
        chain_length=10,
    )

    sa = samplers[sampler_type + "-chunked"]
    hi = sa.hilbert
    ma, w = model_and_weights(hi, sa)

    sampler_state = sa.init_state(ma, w, seed=SAMPLER_SEED)
    sampler_state = sa.reset(ma, w, state=sampler_state)
    samples_ch, sampler_state = sa.sample(
        ma,
        w,
        state=sampler_state,
        chain_length=10,
    )

    np.testing.assert_allclose(samples, samples_ch)
