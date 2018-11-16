import netket as nk
import networkx as nx
import numpy as np
import pytest
from mpi4py import MPI

machines = {}


# TESTS FOR SPIN HILBERT
# Constructing a 1d lattice
g = nk.graph.Hypercube(length=4, ndim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

machines["RbmSpin 1d Hypercube spin"] = [nk.machine.RbmSpin(
    hilbert=hi, alpha=1), hi]

machines["RbmSpinSymm 1d Hypercube spin"] = [nk.machine.RbmSpinSymm(
    hilbert=hi, alpha=2), hi]

machines["Jastrow 1d Hypercube spin"] = [nk.machine.Jastrow(hilbert=hi), hi]

hi = nk.hilbert.Spin(s=0.5, graph=g, total_sz=0)
machines["Jastrow 1d Hypercube spin"] = [
    nk.machine.JastrowSymm(hilbert=hi), hi]


# Layers
act = nk.activation.Lncosh()
layers = [
    nk.layer.FullyConnected(
        input_size=g.n_sites,
        output_size=40,
        activation=act)
]

# this would give a segmentation fault
# layers = [
#    nk.layer.FullyConnected(
# input_size = g.n_sites,
# output_size = 40,
# activation = act)
# ]

# FFNN Machine
machines["FFFN 1d Hypercube spin"] = [nk.machine.FFNN(hi, layers), hi]

machines["MPS Diagonal 1d spin"] = [nk.machine.MPSPeriodicDiagonal(
    hi, bond_dim=3), hi]
machines["MPS 1d spin"] = [nk.machine.MPSPeriodic(hi, bond_dim=3), hi]

# BOSONS
hi = nk.hilbert.Boson(graph=g, n_max=3)
machines["RbmSpin 1d Hypercube boson"] = [nk.machine.RbmSpin(
    hilbert=hi, alpha=1), hi]

machines["RbmSpinSymm 1d Hypercube boson"] = [nk.machine.RbmSpinSymm(
    hilbert=hi, alpha=2), hi]
machines["RbmMultival 1d Hypercube boson"] = [nk.machine.RbmMultival(
    hilbert=hi, n_hidden=10), hi]
machines["Jastrow 1d Hypercube boson"] = [nk.machine.Jastrow(hilbert=hi), hi]

machines["JastrowSymm 1d Hypercube boson"] = [nk.machine.JastrowSymm(
    hilbert=hi), hi]
machines["MPS 1d boson"] = [nk.machine.MPSPeriodic(hi, bond_dim=4), hi]


def log_val(par, machine, v):
    machine.set_parameters(par)
    return machine.log_val(v)


def test_set_get_parameters():
    for name, ma in machines.items():
        print("Machine test: %s" % name)
        machine = ma[0]
        assert(machine.n_par() > 0)
        npar = machine.n_par()
        randpars = np.random.randn(npar) + 1.0j * np.random.randn(npar)
        machine.set_parameters(randpars)
        assert(np.array_equal(machine.get_parameters(), randpars))


import numdifftools as nd
# Ignoring warnings from numdifftools


@pytest.mark.filterwarnings("ignore:`factorial` is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:internal gelsd driver lwork query error:RuntimeWarning")
def test_log_derivative():
    for name, ma in machines.items():
        print("Machine test: %s" % name)
        machine = ma[0]
        npar = machine.n_par()
        randpars = 0.1 * (np.random.randn(npar) + 1.0j * np.random.randn(npar))

        # random visibile state
        hi = ma[1]
        assert(hi.size() > 0)
        rg = nk.RandomEngine(seed=1234)
        v = np.zeros(hi.size())

        for i in range(10):
            hi.random_vals(v, rg)
            grad = (nd.Gradient(log_val))

            machine.set_parameters(randpars)
            assert(np.linalg.norm(machine.der_log(v) -
                                  grad(randpars, machine, v), ord=np.inf) < 1.0e-6)


def test_nvisible():
    for name, ma in machines.items():
        print("Machine test: %s" % name)
        hh = ma[1]
        machine = ma[0]
        assert(machine.n_visible() == hh.size())
