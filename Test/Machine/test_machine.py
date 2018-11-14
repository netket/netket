import netket as nk
import networkx as nx
import numpy as np
from mpi4py import MPI

machines = {}

# TESTS FOR SPIN HILBERT
# Constructing a 1d lattice
g = nk.graph.Hypercube(length=20, ndim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

machines["RbmSpin 1d Hypercube spin"] = nk.machine.RbmSpin(hilbert=hi, alpha=1)

machines["RbmSpinSymm 1d Hypercube spin"] = nk.machine.RbmSpinSymm(
    hilbert=hi, alpha=2)

machines["Jastrow 1d Hypercube spin"] = nk.machine.Jastrow(hilbert=hi)

hi = nk.hilbert.Spin(s=0.5, graph=g, total_sz=0)
machines["Jastrow 1d Hypercube spin"] = nk.machine.JastrowSymm(hilbert=hi)

# Layers
layers = [
    nk.layer.FullyConnected(
        input_size=g.n_sites,
        output_size=40,
        activation=nk.activation.Lncosh())
]

# FFNN Machine
machines["FFFN 1d Hypercube spin"] = nk.machine.FFNN(hi, layers)

machines["MPS Diagonal 1d spin"] = nk.machine.MPSPeriodicDiagonal(
    hi, bond_dim=8)
machines["MPS 1d spin"] = nk.machine.MPSPeriodic(hi, bond_dim=8)


# BOSONS
hi = nk.hilbert.Boson(graph=g, n_max=4)
machines["RbmSpin 1d Hypercube boson"] = nk.machine.RbmSpin(
    hilbert=hi, alpha=1)

machines["RbmSpinSymm 1d Hypercube boson"] = nk.machine.RbmSpinSymm(
    hilbert=hi, alpha=2)
machines["RbmMultival 1d Hypercube boson"] = nk.machine.RbmMultival(
    hilbert=hi, n_hidden=10)
machines["Jastrow 1d Hypercube boson"] = nk.machine.Jastrow(hilbert=hi)

machines["JastrowSymm 1d Hypercube boson"] = nk.machine.JastrowSymm(hilbert=hi)
machines["MPS 1d boson"] = nk.machine.MPSPeriodic(hi, bond_dim=5)


import numpy as np


def test_set_get_parameters():
    for name, ma in machines.items():
        print("Machine test: %s" % name)
        assert(ma.Npar() > 0)
        npar = ma.Npar()
        randpars = np.random.randn(npar) + 1.0j * np.random.randn(npar)
        ma.SetParameters(randpars)
        assert(np.array_equal(ma.GetParameters(), randpars))


def test_log_derivative():
    for name, ma in machines.items():
        print("Machine test: %s" % name)
        ma.InitRandomPars(seed=12, sigma=0.1)
        # TODO maybe use numdifftools for numerical derivatives
        assert(True)


def test_nvisible():
    for name, ma in machines.items():
        print("Machine test: %s" % name)
        # TODO GetHilbert is broken because we return a reference and not a pointer
        # hh=ma.GetHilbert()
        # assert(ma.Nvisible()==ma.GetHilbert().Size())
