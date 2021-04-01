import netket as nk
import networkx as nx
import numpy as np
import pytest
from pytest import approx
from scipy.stats import power_divergence, combine_pvalues, chisquare

import jax
import flax
from jax import numpy as jnp

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

hib = nk.hilbert.Fock(n_max=1, N=g.n_nodes, n_particles=1)

hib_u = nk.hilbert.Fock(n_max=3, N=g.n_nodes)

samplers["Exact: Spin"] = nk.sampler.ExactSampler(hi, n_chains=8)
samplers["Exact: Fock"] = nk.sampler.ExactSampler(hib_u, n_chains=4)

samplers["Metropolis(Local): Spin"] = nk.sampler.MetropolisLocal(hi, n_chains=16)

samplers["MetropolisNumpy(Local): Spin"] = nk.sampler.MetropolisLocalNumpy(
    hi, n_chains=16
)
# samplers["MetropolisNumpy(Local): Fock"] = nk.sampler.MetropolisLocalNumpy(
#    hib_u, n_chains=8
# )
# samplers["MetropolisNumpy(Local): Doubled-Spin"] = nk.sampler.MetropolisLocalNumpy(
#    nk.hilbert.DoubledHilbert(nk.hilbert.Spin(s=0.5, N=2)), n_chains=8
# )

samplers["MetropolisPT(Local): Spin"] = nk.sampler.MetropolisLocalPt(
    hi, n_chains=8, n_replicas=4
)
samplers["MetropolisPT(Local): Fock"] = nk.sampler.MetropolisLocalPt(
    hib_u, n_chains=8, n_replicas=4
)

samplers["Metropolis(Exchange): Fock-1particle)"] = nk.sampler.MetropolisExchange(
    hib, n_chains=16, graph=g
)

samplers["Metropolis(Hamiltonian,Jax): Spin"] = nk.sampler.MetropolisHamiltonian(
    hi,
    hamiltonian=ha,
    reset_chain=True,
)

samplers["Metropolis(Hamiltonian,Numpy): Spin"] = nk.sampler.MetropolisHamiltonianNumpy(
    hi,
    hamiltonian=ha,
    reset_chain=True,
)

samplers["Metropolis(Custom: Sx): Spin"] = nk.sampler.MetropolisCustom(
    hi, move_operators=move_op
)

# samplers["MetropolisPT(Custom: Sx): Spin"] = nk.sampler.MetropolisCustomPt(hi, move_operators=move_op, n_replicas=4)


# The following fixture initialisees a model and it's weights
# for tests that require it.
@pytest.fixture
def rbm_and_weights(request):
    def build_rbm(hilb):
        ma = nk.models.RBM(
            alpha=1,
            dtype=complex,
            kernel_init=nk.nn.initializers.normal(stddev=0.1),
            hidden_bias_init=nk.nn.initializers.normal(stddev=0.1),
        )
        # init network
        w = ma.init(jax.random.PRNGKey(WEIGHT_SEED), jnp.zeros((1, hi.size)))

        return ma, w

    # Do something with the data
    return build_rbm


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
            return sampler.replace(machine_pow=request.param)
        elif cmdline_mpow == "single":
            # samee sampler leads to same rng
            rng = np.random.default_rng(abs(hash((type(sampler), repr(sampler)))))
            exponent = rng.integers(1, 3)  # 1 or 2
            if request.param == exponent:
                return sampler.replace(machine_pow=exponent)
            else:
                pytest.skip(
                    "Running only 1 pdf exponent per sampler. Use --mpow=all to run all pdf exponents."
                )
        elif int(cmdline_mpow) == request.param:
            return sampler.replace(machine_pow=request.param)
        else:
            pytest.skip(f"Running only --mpow={cmdline_mpow}.")

    return fun


def test_states_in_hilbert(sampler, rbm_and_weights):
    hi = sampler.hilbert
    all_states = hi.all_states()

    ma, w = rbm_and_weights(hi)

    for sample in nk.sampler.samples(sampler, ma, w, chain_length=50):
        assert sample.shape == (sampler.n_chains, hi.size)
        for v in sample:
            assert v in all_states

    # if hasattr(sa, "acceptance"):
    #    assert np.min(sampler.acceptance) >= 0 and np.max(sampler.acceptance) <= 1.0


def findrng(rng):
    if hasattr(rng, "_bit_generator"):
        return rng._bit_generator.state["state"]
    else:
        return rng


# Mark tests that we know are ffailing on correctedness
def failing_test(sampler):
    if isinstance(sampler, nk.sampler.MetropolisSampler):
        if isinstance(sampler, nk.sampler.MetropolisPtSampler):
            return True

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
def test_correct_sampling(sampler_c, rbm_and_weights, set_pdf_power):
    sampler = set_pdf_power(sampler_c)

    hi = sampler.hilbert
    all_states = hi.all_states()
    n_states = hi.n_states

    ma, w = rbm_and_weights(hi)

    n_samples = max(40 * n_states, 100)

    ps = np.absolute(nk.nn.to_array(hi, ma, w, normalize=False)) ** sampler.machine_pow
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


def test_throwing(rbm_and_weights):
    with pytest.raises(TypeError):
        nk.sampler.MetropolisHamiltonian(
            hi,
            hamiltonian=10,
            reset_chain=True,
        )

    with pytest.raises(ValueError):
        sampler = nk.sampler.MetropolisHamiltonian(
            nk.hilbert.DoubledHilbert(hi),
            hamiltonian=ha,
            reset_chain=True,
        )

        ma, w = rbm_and_weights(hi)

        sampler_state = sampler.init_state(ma, w, seed=SAMPLER_SEED)

    with pytest.raises(ValueError):
        sampler = nk.sampler.MetropolisHamiltonianNumpy(
            nk.hilbert.Fock(3) ** hi.size,
            hamiltonian=ha,
            reset_chain=True,
        )

        ma, w = rbm_and_weights(hi)

        sampler_state = sampler.init_state(ma, w, seed=SAMPLER_SEED)

    with pytest.raises(flax.errors.ScopeParamShapeError):
        sampler = nk.sampler.MetropolisHamiltonianNumpy(
            nk.hilbert.DoubledHilbert(hi),
            hamiltonian=ha,
            reset_chain=True,
        )

        ma, w = rbm_and_weights(hi)

        sampler_state = sampler.init_state(ma, w, seed=SAMPLER_SEED)
