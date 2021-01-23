import netket as nk
import networkx as nx
import numpy as np
import pytest
from pytest import approx
from scipy.stats import power_divergence, combine_pvalues, chisquare

import jax
from jax import numpy as jnp

samplers = {}

np.random.seed(1234)

from netket.utils import jax_available as test_jax


# TESTS FOR SPIN HILBERT
# Constructing a 1d lattice
g = nk.graph.Hypercube(length=4, n_dim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

# ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)

samplers["Exact RbmSpin"] = nk.sampler.ExactSampler(hi, sample_size=8)

samplers["MetropolisLocal"] = nk.sampler.MetropolisLocal(hi, n_chains=16)

samplers["MetropolisLocal Doubled"] = nk.sampler.MetropolisLocal(
    nk.hilbert.DoubledHilbert(nk.hilbert.Spin(s=0.5, N=2)), n_chains=8
)

hib = nk.hilbert.Fock(n_max=1, N=g.n_nodes, n_particles=1)
sa = nk.sampler.MetropolisExchange(hi, n_chains=16, graph=g)
samplers["MetropolisExchange RbmSpin(fock)"] = sa


# sa = nk.sampler.MetropolisLocalPt(machine=ma, n_replicas=4)
# samplers["MetropolisLocalPt RbmSpin"] = sa

ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
sa = nk.sampler.MetropolisHamiltonian(hi, hamiltonian=ha)
samplers["MetropolisHamiltonian RbmSpin"] = sa

# Test with uniform probability
# maz = nk.machine.RbmSpin(hilbert=hi, alpha=1)
# maz.init_random_parameters(sigma=0)
# sa = nk.sampler.MetropolisLocal(machine=maz, sweep_size=hi.size + 1, n_chains=2)
# samplers["MetropolisLocal RbmSpin ZeroPars"] = sa

# mas = nk.machine.RbmSpinSymm(hilbert=hi, alpha=1, automorphisms=g)
# mas.init_random_parameters(sigma=0.2)
# sa = nk.sampler.MetropolisHamiltonianPt(machine=mas, hamiltonian=ha, n_replicas=4)
# samplers["MetropolisHamiltonianPt RbmSpinSymm"] = sa

hib = nk.hilbert.Fock(N=g.n_nodes, n_max=3)
# ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
# ma.init_random_parameters(sigma=0.1)
# sa =
samplers["MetropolisLocal Boson"] = nk.sampler.MetropolisLocal(hib)

# sa = nk.sampler.MetropolisLocalPt(machine=ma, n_replicas=2)
# samplers["MetropolisLocalPt Boson"] = sa

# hi = nk.hilbert.Boson(N=g.n_nodes, n_max=3)
# ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
# ma.init_random_parameters(sigma=0.1)
samplers["Exact Boson"] = nk.sampler.ExactSampler(hib)


# hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
# g = nk.graph.Hypercube(length=3, n_dim=1)
# ma = nk.machine.RbmSpinSymm(hilbert=hi, alpha=1, automorphisms=g)
# ma.init_random_parameters(sigma=0.2)
# l = hi.size
# X = [[0, 1], [1, 0]]
#
# move_op = nk.operator.LocalOperator(
#    hilbert=hi, operators=[X] * l, acting_on=[[i] for i in range(l)]
# )
#
#
# sa = nk.sampler.CustomSampler(machine=ma, move_operators=move_op)
# samplers["CustomSampler Spin"] = sa
#
#
# sa = nk.sampler.CustomSamplerPt(machine=ma, move_operators=move_op, n_replicas=4)
# samplers["CustomSamplerPt Spin"] = sa
#
# Two types of custom moves
# single spin flips and nearest-neighbours exchanges
# spsm = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
#
# ops = [X] * l
# ops += [spsm] * l
#
# acting_on = [[i] for i in range(l)]
# acting_on += [[i, (i + 1) % l] for i in range(l)]
#
# move_op = nk.operator.LocalOperator(hilbert=hi, operators=ops, acting_on=acting_on)
#
# sa = nk.sampler.CustomSampler(machine=ma, move_operators=move_op)
# samplers["CustomSampler Spin 2 moves"] = sa

# Diagonal density matrix sampling
# ma = nk.machine.density_matrix.RbmSpin(
#    hilbert=hi,
#    alpha=1,
#    use_visible_bias=True,
#    use_hidden_bias=True,
# )
# ma.init_random_parameters(sigma=0.2)
# dm = ma.diagonal()
# sa = nk.sampler.MetropolisLocal(machine=dm)
# samplers["Diagonal Density Matrix"] = sa
#
# sa = nk.sampler.ExactSampler(machine=dm)
# samplers["Exact Diagonal Density Matrix"] = sa
#
# g = nk.graph.Hypercube(length=3, n_dim=1)
# hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
# ma = nk.machine.density_matrix.RbmSpin(
#    hilbert=hi,
#    alpha=1,
#    use_visible_bias=True,
#    use_hidden_bias=True,
# )
#
# ma.init_random_parameters(sigma=0.2)
# samplers["Metropolis Density Matrix"] = nk.sampler.MetropolisLocal(ma, n_chains=16)
#
# sa = nk.sampler.ExactSampler(machine=ma, sample_size=8)
# samplers["Exact Density Matrix"] = sa

# if test_jax:
#    ma = nk.machine.density_matrix.NdmSpinPhase(hilbert=hi, alpha=1, beta=1)
#    ma.init_random_parameters(sigma=0.2)
#    samplers["Metropolis Density Matrix Jax"] = nk.sampler.MetropolisLocal(
#        ma, n_chains=16
#    )

#    ma = nk.machine.JaxRbm(hilbert=hi, alpha=1)
#    ma.init_random_parameters(sigma=0.2)
#    samplers["Metropolis Rbm Jax"] = nk.sampler.MetropolisLocal(ma, n_chains=16)

#    hib = nk.hilbert.Boson(n_max=1, N=g.n_nodes, n_bosons=1)
#    mab = nk.machine.JaxRbm(hilbert=hib, alpha=1)
#    mab.init_random_parameters(sigma=0.2)
#    sa = nk.sampler.MetropolisExchange(machine=mab, n_chains=16, graph=g)
#    samplers["MetropolisExchange RbmSpin(boson) Jax"] = sa

#    # Test a machine which only works with 2D output and not 1D
#    import jax
#    from jax.nn.initializers import glorot_normal

#    def Jastrow(W_init=glorot_normal()):
#        def init_fun(rng, input_shape):
#            N = input_shape[-1]
#            return input_shape[:-1], W_init(rng, (N, N))

#        def apply_fun(W, x, **kwargs):
#            return jax.vmap(
#                lambda W, x: jax.numpy.einsum("i,ij,j", x, W, x), in_axes=(None, 0)
#            )(W, x)

#        return init_fun, apply_fun

#    ma = nk.machine.Jax(hi, Jastrow(), dtype=float)
#    ma.init_random_parameters(sigma=0.2)
#    samplers["Metropolis Jastrow Jax"] = nk.sampler.MetropolisLocal(ma, n_chains=16)
@pytest.fixture
def rbm_and_weights(request):
    def build_rbm(hilb):
        ma = nk.models.RBM(alpha=1, dtype=np.complex64)
        # init network
        w = ma.init(jax.random.PRNGKey(0), jnp.zeros((1, hi.size)))

        return ma, w

    # Do something with the data
    return build_rbm


@pytest.mark.parametrize(
    "sa", [pytest.param(sa, id=name) for name, sa in samplers.items()]
)
def test_states_in_hilbert(sa, rbm_and_weights):
    hi = sa.hilbert
    all_states = hi.all_states()

    ma, w = rbm_and_weights(hi)

    for sample in nk.sampler.samples(sa, ma, w, chain_length=50):
        assert sample.shape == (sa.n_chains, hi.size)
        for v in sample:
            assert v in all_states

    # if hasattr(sa, "acceptance"):
    #    assert np.min(sa.acceptance) >= 0 and np.max(sa.acceptance) <= 1.0


# Testing that samples generated from direct sampling are compatible with those
# generated by markov chain sampling
# here we use a combination of power divergence tests


def is_metropolis_ham(sa):
    print(type(sa))
    if hasattr(sa, "rule"):
        print("hello")
        if isinstance(sa.rule, nk.sampler.rules.HamiltonianRule):
            return True
    return False


@pytest.mark.parametrize(
    "sa", [pytest.param(sa, id=name) for name, sa in samplers.items()]
)
def test_correct_sampling(sa, rbm_and_weights):

    if is_metropolis_ham(sa):
        pytest.skip("it works but it's too slow to test")

    hi = sa.hilbert
    all_states = hi.all_states()

    ma, w = rbm_and_weights(hi)

    n_states = hi.n_states

    n_samples = max(40 * n_states, 100)

    ord = np.random.randint(1, 3, size=()).item()
    assert ord == 1 or ord == 2

    sa = sa.replace(machine_pow=ord)

    ps = np.absolute(nk.nn.to_array(hi, ma, w, normalize=False)) ** ord
    ps /= ps.sum()

    n_rep = 6
    pvalues = np.zeros(n_rep)

    sampler_state = sa.init_state(ma, w, seed=12345)

    for jrep in range(n_rep):
        sampler_state = sa.reset(ma, w, state=sampler_state)

        # Burnout phase
        samples, sampler_state = sa.sample(
            ma, w, state=sampler_state, chain_length=n_samples // 100
        )

        assert samples.shape == (
            n_samples // 100,
            sa.n_chains,
            hi.size,
        )  # sa.sample_shape

        samples, sampler_state = sa.sample(
            ma, w, state=sampler_state, chain_length=n_samples
        )

        assert samples.shape == (n_samples, sa.n_chains, hi.size)  # sa.sample_shape

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
