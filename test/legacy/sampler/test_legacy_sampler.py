import netket.legacy as nk
import networkx as nx
import numpy as np
import pytest
from pytest import approx
from scipy.stats import power_divergence, combine_pvalues, chisquare
from netket.legacy.random import randint

pytestmark = pytest.mark.legacy

samplers = {}

nk.random.seed(1234567)
np.random.seed(1234)

from netket.utils import jax_available as test_jax


# TESTS FOR SPIN HILBERT
# Constructing a 1d lattice
g = nk.graph.Hypercube(length=4, n_dim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
ma.init_random_parameters(sigma=0.2)

sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=16)
samplers["MetropolisLocal RbmSpin"] = sa

hib = nk.hilbert.Boson(n_max=1, N=g.n_nodes, n_bosons=1)
mab = nk.machine.RbmSpin(hilbert=hib, alpha=1)
mab.init_random_parameters(sigma=0.2)
sa = nk.sampler.MetropolisExchange(machine=mab, n_chains=16, graph=g)
samplers["MetropolisExchange RbmSpin(boson)"] = sa

sa = nk.sampler.ExactSampler(machine=ma, sample_size=8)
samplers["Exact RbmSpin"] = sa

sa = nk.sampler.MetropolisLocalPt(machine=ma, n_replicas=4)
samplers["MetropolisLocalPt RbmSpin"] = sa

ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha)
samplers["MetropolisHamiltonian RbmSpin"] = sa

# Test with uniform probability
maz = nk.machine.RbmSpin(hilbert=hi, alpha=1)
maz.init_random_parameters(sigma=0)
sa = nk.sampler.MetropolisLocal(machine=maz, sweep_size=hi.size + 1, n_chains=2)
samplers["MetropolisLocal RbmSpin ZeroPars"] = sa

mas = nk.machine.RbmSpinSymm(hilbert=hi, alpha=1, automorphisms=g)
mas.init_random_parameters(sigma=0.2)
sa = nk.sampler.MetropolisHamiltonianPt(machine=mas, hamiltonian=ha, n_replicas=4)
samplers["MetropolisHamiltonianPt RbmSpinSymm"] = sa

hi = nk.hilbert.Boson(N=g.n_nodes, n_max=3)
ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
ma.init_random_parameters(sigma=0.1)
sa = nk.sampler.MetropolisLocal(machine=ma)
samplers["MetropolisLocal Boson"] = sa

sa = nk.sampler.MetropolisLocalPt(machine=ma, n_replicas=2)
samplers["MetropolisLocalPt Boson"] = sa

hi = nk.hilbert.Boson(N=g.n_nodes, n_max=3)
ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
ma.init_random_parameters(sigma=0.1)
sa = nk.sampler.ExactSampler(machine=ma)
samplers["Exact Boson"] = sa

hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
g = nk.graph.Hypercube(length=3, n_dim=1)
ma = nk.machine.RbmSpinSymm(hilbert=hi, alpha=1, automorphisms=g)
ma.init_random_parameters(sigma=0.2)
l = hi.size
X = [[0, 1], [1, 0]]

move_op = nk.operator.LocalOperator(
    hilbert=hi, operators=[X] * l, acting_on=[[i] for i in range(l)]
)


sa = nk.sampler.CustomSampler(machine=ma, move_operators=move_op)
samplers["CustomSampler Spin"] = sa


sa = nk.sampler.CustomSamplerPt(machine=ma, move_operators=move_op, n_replicas=4)
samplers["CustomSamplerPt Spin"] = sa

# Two types of custom moves
# single spin flips and nearest-neighbours exchanges
spsm = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]

ops = [X] * l
ops += [spsm] * l

acting_on = [[i] for i in range(l)]
acting_on += [[i, (i + 1) % l] for i in range(l)]

move_op = nk.operator.LocalOperator(hilbert=hi, operators=ops, acting_on=acting_on)

sa = nk.sampler.CustomSampler(machine=ma, move_operators=move_op)
samplers["CustomSampler Spin 2 moves"] = sa

# Diagonal density matrix sampling
ma = nk.machine.density_matrix.RbmSpin(
    hilbert=hi,
    alpha=1,
    use_visible_bias=True,
    use_hidden_bias=True,
)
ma.init_random_parameters(sigma=0.2)
dm = ma.diagonal()
sa = nk.sampler.MetropolisLocal(machine=dm)
samplers["Diagonal Density Matrix"] = sa

sa = nk.sampler.ExactSampler(machine=dm)
samplers["Exact Diagonal Density Matrix"] = sa

g = nk.graph.Hypercube(length=3, n_dim=1)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
ma = nk.machine.density_matrix.RbmSpin(
    hilbert=hi,
    alpha=1,
    use_visible_bias=True,
    use_hidden_bias=True,
)

ma.init_random_parameters(sigma=0.2)
samplers["Metropolis Density Matrix"] = nk.sampler.MetropolisLocal(ma, n_chains=16)

sa = nk.sampler.ExactSampler(machine=ma, sample_size=8)
samplers["Exact Density Matrix"] = sa

if test_jax:
    ma = nk.machine.density_matrix.NdmSpinPhase(hilbert=hi, alpha=1, beta=1)
    ma.init_random_parameters(sigma=0.2)
    samplers["Metropolis Density Matrix Jax"] = nk.sampler.MetropolisLocal(
        ma, n_chains=16
    )

    ma = nk.machine.JaxRbm(hilbert=hi, alpha=1)
    ma.init_random_parameters(sigma=0.2)
    samplers["Metropolis Rbm Jax"] = nk.sampler.MetropolisLocal(ma, n_chains=16)

    hib = nk.hilbert.Boson(n_max=1, N=g.n_nodes, n_bosons=1)
    mab = nk.machine.JaxRbm(hilbert=hib, alpha=1)
    mab.init_random_parameters(sigma=0.2)
    sa = nk.sampler.MetropolisExchange(machine=mab, n_chains=16, graph=g)
    samplers["MetropolisExchange RbmSpin(boson) Jax"] = sa

    # Test a machine which only works with 2D output and not 1D
    import jax
    from jax.nn.initializers import glorot_normal

    def Jastrow(W_init=glorot_normal()):
        def init_fun(rng, input_shape):
            N = input_shape[-1]
            return input_shape[:-1], W_init(rng, (N, N))

        def apply_fun(W, x, **kwargs):
            return jax.vmap(
                lambda W, x: jax.numpy.einsum("i,ij,j", x, W, x), in_axes=(None, 0)
            )(W, x)

        return init_fun, apply_fun

    ma = nk.machine.Jax(hi, Jastrow(), dtype=float)
    ma.init_random_parameters(sigma=0.2)
    samplers["Metropolis Jastrow Jax"] = nk.sampler.MetropolisLocal(ma, n_chains=16)


def test_states_in_hilbert():
    for name, sa in samplers.items():
        print("Sampler test: %s" % name)

        ma = sa.machine
        hi = ma.hilbert
        localstates = hi.local_states

        for sample in sa.samples(100):
            assert sample.shape[1] == ma.input_size
            for v in sample.reshape(-1):
                assert v in localstates

        if hasattr(sa, "acceptance"):
            assert np.min(sa.acceptance) >= 0 and np.max(sa.acceptance) <= 1.0


# Testing that samples generated from direct sampling are compatible with those
# generated by markov chain sampling
# here we use a combination of power divergence tests


def test_correct_sampling():
    for name, sa in samplers.items():
        print("Sampler test: %s" % name)

        ma = sa.machine
        hi = ma.hilbert

        if ma.input_size == 2 * hi.size:
            hi = nk.hilbert.DoubledHilbert(hi)
        n_states = hi.n_states

        n_samples = max(40 * n_states, 10000)

        ord = randint(1, 3, size=()).item()
        assert ord == 1 or ord == 2

        sa.machine_pow = ord

        ps = np.absolute(ma.to_array()) ** ord
        ps /= ps.sum()

        n_rep = 6
        pvalues = np.zeros(n_rep)

        sa.reset(True)

        for jrep in range(n_rep):

            # Burnout phase
            samples = sa.generate_samples(n_samples // 10)

            assert (samples.shape[1], samples.shape[2]) == sa.sample_shape

            samples = sa.generate_samples(n_samples)

            assert samples.shape[2] == ma.input_size
            sttn = hi.states_to_numbers(np.asarray(samples.reshape(-1, ma.input_size)))
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
