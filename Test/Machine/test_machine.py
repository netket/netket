import netket as nk
import networkx as nx
import numpy as np
import pytest
from pytest import approx
import os
from netket.hilbert import PySpin as Spin

test_jax = True
try:
    import torch

    test_torch = True
except:
    test_torch = False


def merge_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


machines = {}

# TESTS FOR SPIN HILBERT
# Constructing a 1d lattice
g = nk.graph.Hypercube(length=4, n_dim=1)

# Hilbert space of spins from given graph
hi = Spin(s=0.5, graph=g)


if test_jax:
    import jax
    import jax.experimental
    import jax.experimental.stax

    def initializer(rng, shape):
        return np.random.normal(scale=0.05, size=shape)

    machines["Jax Real"] = nk.machine.Jax(
        hi,
        jax.experimental.stax.serial(
            jax.experimental.stax.Dense(4, initializer, initializer),
            jax.experimental.stax.Relu,
            jax.experimental.stax.Dense(2, initializer, initializer),
            jax.experimental.stax.Relu,
            jax.experimental.stax.Dense(2, initializer, initializer),
        ),
        dtype=float,
    )

    machines["Jax Complex"] = nk.machine.Jax(
        hi,
        jax.experimental.stax.serial(
            jax.experimental.stax.Dense(4, initializer, initializer),
            jax.experimental.stax.Tanh,
            jax.experimental.stax.Dense(2, initializer, initializer),
            jax.experimental.stax.Tanh,
            jax.experimental.stax.Dense(1, initializer, initializer),
        ),
        dtype=complex,
    )


if test_torch:
    import torch

    input_size = hi.size
    alpha = 1

    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, alpha * input_size),
        torch.nn.Sigmoid(),
        torch.nn.Linear(alpha * input_size, 2),
        torch.nn.Sigmoid(),
    )
    machines["Torch"] = nk.machine.Torch(model, hilbert=hi)


machines["RbmSpin 1d Hypercube spin"] = nk.machine.RbmSpin(hilbert=hi, alpha=2)

machines["RbmSpinSymm 1d Hypercube spin"] = nk.machine.RbmSpinSymm(hilbert=hi, alpha=2)

machines["Real RBM"] = nk.machine.RbmSpinReal(hilbert=hi, alpha=2)

machines["Phase RBM"] = nk.machine.RbmSpinPhase(hilbert=hi, alpha=2)

machines["Jastrow 1d Hypercube spin"] = nk.machine.Jastrow(hilbert=hi)

hi = Spin(s=0.5, graph=g, total_sz=0)
machines["Jastrow 1d Hypercube spin symm bias"] = nk.machine.Jastrow(
    hilbert=hi, use_visible_bias=True, symmetry=True
)

dm_machines = {}
# dm_machines["Phase NDM"] = nk.machine.NdmSpinPhase(
#     hilbert=hi,
#     alpha=2,
#     beta=2,
#     use_visible_bias=True,
#     use_hidden_bias=True,
#     use_ancilla_bias=True,
# )

# Layers
layers = (
    nk.layer.FullyConnected(input_size=g.n_sites, output_size=40),
    nk.layer.Lncosh(input_size=40),
)

# FFNN Machine
# machines["FFFN 1d Hypercube spin FullyConnected"] = nk.machine.FFNN(hi, layers)

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
# machines["FFFN 1d Hypercube spin Convolutional Hypercube"] = nk.machine.FFNN(hi, layers)

# machines["MPS Diagonal 1d spin"] = nk.machine.MPSPeriodicDiagonal(hi, bond_dim=3)
# machines["MPS 1d spin"] = nk.machine.MPSPeriodic(hi, bond_dim=3)

# BOSONS
hi = nk.hilbert.Boson(graph=g, n_max=3)
machines["RbmSpin 1d Hypercube boson"] = nk.machine.RbmSpin(hilbert=hi, alpha=1)

machines["RbmSpinSymm 1d Hypercube boson"] = nk.machine.RbmSpinSymm(hilbert=hi, alpha=2)
machines["RbmMultiVal 1d Hypercube boson"] = nk.machine.RbmMultiVal(
    hilbert=hi, n_hidden=2
)
machines["Jastrow 1d Hypercube boson"] = nk.machine.Jastrow(hilbert=hi)

machines["JastrowSymm 1d Hypercube boson real"] = nk.machine.JastrowSymm(
    hilbert=hi, dtype=float
)
machines["MPS 1d boson"] = nk.machine.MPSPeriodic(hi, bond_dim=4)


np.random.seed(12346)


def same_derivatives(der_log, num_der_log, eps=1.0e-6):
    assert der_log.shape == num_der_log.shape
    assert np.max(np.real(der_log - num_der_log)) == approx(0.0, rel=eps, abs=eps)

    # The imaginary part is a bit more tricky, there might be an arbitrary phase shift
    assert np.max(np.exp(np.imag(der_log - num_der_log) * 1.0j) - 1.0) == approx(
        0.0, rel=eps, abs=eps
    )


def log_val_f(par, machine, v):
    machine.parameters = np.copy(par)
    if v.ndim != 1:
        if v.size != v.shape[1]:
            raise RuntimeError(
                "numerical derivatives can be tested only for non batched inputs"
            )

    return machine.log_val(v.reshape(1, -1))[0]


def log_val_vec_f(par, machine, v, vec):
    machine.parameters = np.copy(par)
    if v.ndim == 2:
        out_val = machine.log_val(v)
    else:
        out_val = machine.log_val(v.reshape(1, -1))
    assert vec.shape == out_val.shape
    return np.vdot(out_val, vec)


def central_diff_grad(func, x, eps, *args):
    grad = np.zeros(len(x), dtype=complex)
    epsd = np.zeros(len(x), dtype=complex)
    epsd[0] = eps
    for i in range(len(x)):
        assert not np.any(np.isnan(x + epsd))
        grad[i] = 0.5 * (func(x + epsd, *args) - func(x - epsd, *args))
        assert not np.isnan(grad[i])
        grad[i] /= eps
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
        machine.parameters = np.copy(randpars)
        if machine.is_holomorphic:
            assert np.array_equal(machine.parameters, randpars)
        else:
            assert np.array_equal(machine.parameters.real, randpars.real)

        machine.parameters = np.zeros(npar)
        assert np.count_nonzero(np.abs(machine.parameters)) == 0


def test_save_load_parameters(tmpdir):
    for name, machine in merge_dicts(machines, dm_machines).items():
        print("Machine test: %s" % name)
        assert machine.n_par > 0
        n_par = machine.n_par
        randpars = np.random.randn(n_par) + 1.0j * np.random.randn(n_par)

        machine.parameters = randpars
        fn = tmpdir.mkdir("datawf").join("test.wf")

        filename = os.path.join(fn.dirname, fn.basename)

        machine.save(filename)
        machine.parameters = np.empty(n_par)
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

        v = np.zeros(hi.size)

        for i in range(100):
            hi.random_vals(v)

            randpars = 0.1 * (np.random.randn(npar) + 1.0j * np.random.randn(npar))
            machine.parameters = randpars

            der_log = machine.der_log(v.reshape((1, -1))).reshape(-1)

            if "Jastrow" in name:
                assert np.max(np.imag(der_log)) == approx(0.0)

            num_der_log = central_diff_grad(log_val_f, randpars, 1.0e-9, machine, v)

            same_derivatives(der_log, num_der_log)
            # print(np.linalg.norm(der_log - num_der_log))
            # Check if machine is correctly set to be holomorphic
            # The check is done only on smaller subset of parameters, for speed
            if i % 10 == 0 and machine.is_holomorphic:
                check_holomorphic(log_val_f, randpars, 1.0e-8, machine, v)


def test_vector_jacobian():
    for name, machine in merge_dicts(machines, dm_machines).items():
        print("Machine test: %s" % name)

        npar = machine.n_par

        # random visibile state
        hi = machine.hilbert
        assert hi.size > 0

        batch_size = 100
        v = np.zeros((batch_size, hi.size))

        for i in range(batch_size):
            hi.random_vals(v[i])

        machine.init_random_parameters(seed=1234, sigma=0.1)
        randpars = machine.parameters

        vec = np.random.uniform(size=batch_size) + 1.0j * np.random.uniform(
            size=batch_size
        ) / float(batch_size)

        vjp = np.zeros(machine.n_par, dtype=np.complex128)
        machine.vector_jacobian_prod(v, vec, vjp)

        num_der_log = central_diff_grad(
            log_val_vec_f, randpars, 1.0e-6, machine, v, vec
        )
        print(np.max(vjp.imag - num_der_log.imag))
        print(np.max(vjp.real - num_der_log.real))
        same_derivatives(vjp, num_der_log)


def test_nvisible():
    for name, machine in machines.items():
        print("Machine test: %s" % name)
        hi = machine.hilbert

        assert machine.n_visible == hi.size

    for name, machine in dm_machines.items():
        print("Machine test: %s" % name)
        hip = machine.hilbert_physical
        hi = machine.hilbert

        assert machine.n_visible_physical * 2 == machine.n_visible
        assert machine.n_visible_physical == hip.size
        assert machine.n_visible == hi.size


def test_dm_batched():
    for name, machine in dm_machines.items():
        print("Machine test: %s" % name)

        npar = machine.n_par
        randpars = 0.5 * (np.random.randn(npar) + 1.0j * np.random.randn(npar))
        machine.parameters = randpars

        hi = machine.hilbert

        rg = nk.utils.RandomEngine(seed=1234)

        N_batches = 100
        states = np.zeros((N_batches, hi.size))

        # loop over different random states
        for i in range(100):

            # generate a random state
            rstate = np.zeros(hi.size)
            hi.random_vals(rstate)
            states[i, :] = rstate

        log_val_batch = machine.log_val(states)
        der_log_batch = machine.der_log(states)

        for i in range(100):

            val = machine.log_val(states[i, :])
            der = machine.der_log(states[i, :])

            assert np.max(np.abs(val - log_val_batch[i])) == approx(0.0)
            same_derivatives(der, der_log_batch[i])
            # The imaginary part is a bit more tricky, there might be an arbitrary phase shift


def test_to_array():
    for name, machine in machines.items():
        print("Machine test: %s" % name)

        npar = machine.n_par

        randpars = 0.5 * (np.random.randn(npar) + 1.0j * np.random.randn(npar))
        if "Torch" in name:
            randpars = randpars.real
        machine.parameters = randpars

        hi = machine.hilbert

        all_psis = machine.to_array(normalize=False)
        # test shape
        assert all_psis.shape[0] == hi.n_states
        assert len(all_psis.shape) == 1

        log_vals = machine.log_val(hi.all_states())
        logmax = log_vals.real.max()

        norm = (np.abs(np.exp(log_vals - logmax)) ** 2).sum()

        # test random values
        for i in range(100):
            rstate = np.zeros(hi.size)
            local_states = hi.local_states
            hi.random_vals(rstate)

            number = hi.state_to_number(rstate)

            assert np.abs(
                np.exp(machine.log_val(rstate.reshape(1, -1)) - logmax)
                - all_psis[number]
            ) == approx(0.0)

        # test rescale
        all_psis_normalized = machine.to_array(normalize=True)

        # test random values
        for i in range(100):
            rstate = np.zeros(hi.size)
            local_states = hi.local_states
            hi.random_vals(rstate)

            number = hi.state_to_number(rstate)

            assert np.abs(
                np.exp(machine.log_val(rstate.reshape(1, -1)) - logmax) / np.sqrt(norm)
                - all_psis_normalized[number]
            ) == approx(0.0)
