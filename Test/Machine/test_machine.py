import netket as nk
import networkx as nx
import numpy as np
from mpi4py import MPI

machines = {}

# TESTS FOR SPIN HILBERT
# Constructing a 1d lattice
g = nk.graph.Hypercube(length=20, ndim=1)

# Hilbert space of spins from given graph
hi1 = nk.hilbert.Spin(s=0.5, graph=g)

machines["RbmSpin 1d Hypercube spin"] = nk.machine.RbmSpin(
    hilbert=hi1, alpha=1)

machines["RbmSpinSymm 1d Hypercube spin"] = nk.machine.RbmSpinSymm(
    hilbert=hi1, alpha=2)

machines["Jastrow 1d Hypercube spin"] = nk.machine.Jastrow(hilbert=hi1)

hi2 = nk.hilbert.Spin(s=0.5, graph=g, total_sz=0)
machines["Jastrow 1d Hypercube spin"] = nk.machine.JastrowSymm(hilbert=hi2)

# Layers
layers = [
    nk.layer.FullyConnected(
        input_size=g.n_sites,
        output_size=40,
        activation=nk.activation.Lncosh())
]

# FFNN Machine
machines["FFFN 1d Hypercube spin"] = nk.machine.FFNN(hi2, layers)

machines["MPS Diagonal 1d spin"] = nk.machine.MPSPeriodicDiagonal(
    hi2, bond_dim=8)
machines["MPS 1d spin"] = nk.machine.MPSPeriodic(hi2, bond_dim=8)

# BOSONS
hi3 = nk.hilbert.Boson(graph=g, n_max=4)
machines["RbmSpin 1d Hypercube boson"] = nk.machine.RbmSpin(
    hilbert=hi3, alpha=1)

machines["RbmSpinSymm 1d Hypercube boson"] = nk.machine.RbmSpinSymm(
    hilbert=hi3, alpha=2)
machines["RbmMultival 1d Hypercube boson"] = nk.machine.RbmMultival(
    hilbert=hi3, n_hidden=10)
machines["Jastrow 1d Hypercube boson"] = nk.machine.Jastrow(hilbert=hi3)

machines["JastrowSymm 1d Hypercube boson"] = nk.machine.JastrowSymm(
    hilbert=hi3)
machines["MPS 1d boson"] = nk.machine.MPSPeriodic(hi3, bond_dim=5)


def log_val(par, machine, v):
    machine.set_parameters(par)
    return machine.log_val(v)


import numdifftools as nd


def test_set_get_parameters():
    for name, ma in machines.items():
        print("Machine test: %s" % name)
        assert(ma.n_par() > 0)
        npar = ma.n_par()
        randpars = np.random.randn(npar) + 1.0j * np.random.randn(npar)
        ma.set_parameters(randpars)
        assert(np.array_equal(ma.get_parameters(), randpars))


def test_log_derivative():
    for name, ma in machines.items():
        print("Machine test: %s" % name)
        npar = ma.n_par()
        randpars = np.random.randn(npar) + 1.0j * np.random.randn(npar)

        # random visibile state
        hi = ma.get_hilbert()
        assert(hi.size() > 0)
        rg = nk.RandomEngine(seed=1234)
        v = np.zeros(hi.size())
        hi.random_vals(v, rg)

        grad = (nd.Gradient(log_val))

        ma.set_parameters(randpars)
        assert(np.linalg.norm(ma.der_log(v) -
                              grad(randpars, ma, v), ord=np.inf) < 1.0e-6)


def test_nvisible():
    for name, ma in machines.items():
        print("Machine test: %s" % name)
        hh = ma.get_hilbert()
        assert(ma.n_visible() == ma.get_hilbert().size())
