import netket as nk
import networkx as nx
import numpy as np
import pytest
from pytest import approx
import os

from rbm import PyRbm

def merge_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


machines = {}

# TESTS FOR SPIN HILBERT
# Constructing a 1d lattice
g = nk.graph.Hypercube(length=4, n_dim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

machines["RbmSpin 1d Hypercube spin"] = nk.machine.RbmSpin(hilbert=hi, alpha=2)

machines["PyRbm 1d Hypercube spin"] = PyRbm(hilbert=hi, alpha=3)

machines["RbmSpinSymm 1d Hypercube spin"] = nk.machine.RbmSpinSymm(hilbert=hi, alpha=2)

machines["Real RBM"] = nk.machine.RbmSpinReal(hilbert=hi, alpha=1)

machines["Phase RBM"] = nk.machine.RbmSpinPhase(hilbert=hi, alpha=2)

machines["Jastrow 1d Hypercube spin"] = nk.machine.Jastrow(hilbert=hi)

hi = nk.hilbert.Spin(s=0.5, graph=g, total_sz=0)
machines["Jastrow 1d Hypercube spin"] = nk.machine.JastrowSymm(hilbert=hi)

dm_machines = {}
dm_machines["Phase NDM"] = nk.machine.NdmSpinPhase(
    hilbert=hi,
    alpha=2,
    beta=2,
    use_visible_bias=True,
    use_hidden_bias=True,
    use_ancilla_bias=True,
)

# Layers
layers = (
    nk.layer.FullyConnected(input_size=g.n_sites, output_size=40),
    nk.layer.Lncosh(input_size=40),
)

# FFNN Machine
machines["FFFN 1d Hypercube spin FullyConnected"] = nk.machine.FFNN(hi, layers)

layers = (
    nk.layer.ConvolutionalHypercube(
        length=4,
        n_dim=1,
        input_channels=1,
        output_channels=2,
        stride=1,
        kernel_length=2,
        use_bias=True,
    ),
    nk.layer.Lncosh(input_size=8),
)

# FFNN Machine
machines["FFFN 1d Hypercube spin Convolutional Hypercube"] = nk.machine.FFNN(hi, layers)

machines["MPS Diagonal 1d spin"] = nk.machine.MPSPeriodicDiagonal(hi, bond_dim=3)
machines["MPS 1d spin"] = nk.machine.MPSPeriodic(hi, bond_dim=3)

# BOSONS
hi = nk.hilbert.Boson(graph=g, n_max=3)
machines["RbmSpin 1d Hypercube boson"] = nk.machine.RbmSpin(hilbert=hi, alpha=1)

machines["RbmSpinSymm 1d Hypercube boson"] = nk.machine.RbmSpinSymm(hilbert=hi, alpha=2)
machines["RbmMultiVal 1d Hypercube boson"] = nk.machine.RbmMultiVal(
    hilbert=hi, n_hidden=10
)
machines["Jastrow 1d Hypercube boson"] = nk.machine.Jastrow(hilbert=hi)

machines["JastrowSymm 1d Hypercube boson"] = nk.machine.JastrowSymm(hilbert=hi)
machines["MPS 1d boson"] = nk.machine.MPSPeriodic(hi, bond_dim=4)


np.random.seed(12346)


def same_derivatives(der_log, num_der_log, eps=1.0e-6):
    assert np.max(np.real(der_log - num_der_log)) == approx(0.0, rel=eps, abs=eps)
    # The imaginary part is a bit more tricky, there might be an arbitrary phase shift
    assert np.max(np.exp(np.imag(der_log - num_der_log) * 1.0j) - 1.0) == approx(
        0.0, rel=eps, abs=eps
    )


def log_val_f(par, machine, v):
    machine.parameters = np.copy(par)
    return machine.log_val(v)


def central_diff_grad(func, x, eps, *args):
    grad = np.zeros(len(x), dtype=complex)
    epsd = np.zeros(len(x), dtype=complex)
    epsd[0] = eps
    for i in range(len(x)):
        grad[i] = 0.5 * (func(x + epsd, *args) - func(x - epsd, *args)) / eps
        epsd = np.roll(epsd, 1)
    return grad


def check_holomorphic(func, x, eps, *args):
    gr = central_diff_grad(func, x, eps, *args)
    gi = central_diff_grad(func, x, eps * 1.0j, *args)
    same_derivatives(gr, gi)


def test_set_get_parameters():
    for name, machine in merge_dicts(machines, dm_machines).items():
        print("Machine test: %s" % name)
        assert machine.n_par > 0
        npar = machine.n_par
        randpars = np.random.randn(npar) + 1.0j * np.random.randn(npar)
        machine.parameters = randpars
        if machine.is_holomorphic:
            assert np.array_equal(machine.parameters, randpars)
        else:
            assert np.array_equal(machine.parameters.real, randpars.real)


def test_save_load_parameters(tmpdir):
    for name, machine in merge_dicts(machines, dm_machines).items():
        print("Machine test: %s" % name)
        assert machine.n_par > 0
        n_par = machine.n_par
        randpars = np.random.randn(n_par) + 1.0j * np.random.randn(n_par)

        machine.parameters = np.copy(randpars)
        fn = tmpdir.mkdir("datawf").join("test.wf")

        filename = os.path.join(fn.dirname, fn.basename)

        machine.save(filename)
        machine.parameters = np.zeros(n_par, dtype=complex)
        machine.load(filename)
        os.remove(filename)
        os.rmdir(fn.dirname)
        if machine.is_holomorphic:
            assert np.array_equal(machine.parameters, randpars)
        else:
            assert np.array_equal(machine.parameters.real, randpars.real)


def test_log_derivative():
    for name, machine in merge_dicts(machines, dm_machines).items():
        print("Machine test: %s" % name)

        npar = machine.n_par

        # random visibile state
        hi = machine.hilbert
        assert hi.size > 0
        rg = nk.utils.RandomEngine(seed=1234)
        v = np.zeros(hi.size)

        for i in range(100):
            hi.random_vals(v, rg)

            randpars = 0.1 * (np.random.randn(npar) + 1.0j * np.random.randn(npar))
            machine.parameters = randpars
            der_log = machine.der_log(v)

            if "Jastrow" in name:
                assert np.max(np.imag(der_log)) == approx(0.0)

            num_der_log = central_diff_grad(log_val_f, randpars, 1.0e-8, machine, v)

            same_derivatives(der_log, num_der_log)

            # Check if machine is correctly set to be holomorphic
            # The check is done only on smaller subset of parameters, for speed
            if i % 10 == 0 and machine.is_holomorphic:
                check_holomorphic(log_val_f, randpars, 1.0e-8, machine, v)


def test_log_val_diff():
    for name, machine in merge_dicts(machines, dm_machines).items():
        print("Machine test: %s" % name)

        npar = machine.n_par
        randpars = 0.5 * (np.random.randn(npar) + 1.0j * np.random.randn(npar))
        machine.parameters = randpars

        hi = machine.hilbert

        rg = nk.utils.RandomEngine(seed=1234)

        # loop over different random states
        for i in range(100):

            # generate a random state
            rstate = np.zeros(hi.size)
            local_states = hi.local_states
            hi.random_vals(rstate, rg)

            tochange = []
            newconfs = []

            # random number of changes
            for i in range(100):
                n_change = np.random.randint(low=0, high=hi.size)
                # generate n_change unique sites to be changed
                tochange.append(np.random.choice(hi.size, n_change, replace=False))
                newconfs.append(np.random.choice(local_states, n_change))

            ldiffs = machine.log_val_diff(rstate, tochange, newconfs)
            valzero = machine.log_val(rstate)

            for toc, newco, ldiff in zip(tochange, newconfs, ldiffs):
                rstatet = np.array(rstate)

                for newc in newco:
                    assert newc in local_states

                for t in toc:
                    assert t >= 0 and t < hi.size

                assert len(toc) == len(newco)

                if len(toc) == 0:
                    assert ldiff == approx(0.0)

                hi.update_conf(rstatet, toc, newco)
                ldiff_num = machine.log_val(rstatet) - valzero

                assert np.max(np.real(ldiff_num - ldiff)) == approx(0.0)
                # The imaginary part is a bit more tricky, there might be an arbitrary phase shift
                assert np.max(np.exp(np.imag(ldiff_num - ldiff) * 1.0j)) == approx(1.0)
                assert np.min(np.exp(np.imag(ldiff_num - ldiff) * 1.0j)) == approx(1.0)


def test_nvisible():
    for name, machine in machines.items():
        print("Machine test: %s" % name)
        hi = machine.hilbert

        assert machine.n_visible == hi.size

    for name, machine in dm_machines.items():
        print("Machine test: %s" % name)
        hip = machine.hilbert_physical
        hi = machine.hilbert

        assert machine.n_visible == hip.size
        assert machine.n_visible * 2 == hi.size
