import netket.legacy as nk
import networkx as nx
import numpy as np
import pytest
from pytest import approx
import os
from netket.hilbert import Spin

test_jax = nk.utils.jax_available
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
dm_machines = {}

# TESTS FOR SPIN HILBERT
# Constructing a 1d lattice
g = nk.graph.Hypercube(length=4, n_dim=1)

# Hilbert space of spins from given graph
hi = Spin(s=0.5, N=g.n_nodes)


if test_jax:
    import jax
    import jax.experimental
    import jax.experimental.stax

    def initializer(rng, shape):
        return np.random.normal(scale=0.05, size=shape)

    # machines["Jax Real"] = nk.machine.Jax(
    #     hi,
    #     jax.experimental.stax.serial(
    #         jax.experimental.stax.Dense(4, initializer, initializer),
    #         jax.experimental.stax.Relu,
    #         jax.experimental.stax.Dense(2, initializer, initializer),
    #         jax.experimental.stax.Relu,
    #         jax.experimental.stax.Dense(2, initializer, initializer),
    #     ),
    #     dtype=float,
    # )

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

    machines["Jax mps"] = nk.machine.MPSPeriodic(hi, graph=g, bond_dim=4, dtype=complex)

    machines["Jax RbmSpinPhase (R->C)"] = nk.machine.JaxRbmSpinPhase(hi, alpha=1)

    dm_machines["Jax NDM"] = nk.machine.density_matrix.NdmSpinPhase(hi, alpha=1, beta=1)


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

machines["RbmSpinSymm 1d Hypercube spin"] = nk.machine.RbmSpinSymm(
    hilbert=hi, alpha=2, automorphisms=g
)

machines["Real RBM"] = nk.machine.RbmSpinReal(hilbert=hi, alpha=2)

machines["Phase RBM"] = nk.machine.RbmSpinPhase(hilbert=hi, alpha=2)

machines["Jastrow 1d Hypercube spin"] = nk.machine.Jastrow(hilbert=hi)

hi = Spin(s=0.5, N=g.n_nodes, total_sz=0)
machines["Jastrow 1d Hypercube spin symm bias"] = nk.machine.Jastrow(
    hilbert=hi, use_visible_bias=True, automorphisms=g
)

dm_machines["Phase NDM"] = nk.machine.density_matrix.RbmSpin(
    hilbert=hi,
    alpha=2,
    use_visible_bias=True,
    use_hidden_bias=True,
)


# machines["MPS Diagonal 1d spin"] = nk.machine.MPSPeriodicDiagonal(hi, bond_dim=3)
# machines["MPS 1d spin"] = nk.machine.MPSPeriodic(hi, bond_dim=3)

# BOSONS
hi = nk.hilbert.Boson(N=g.n_nodes, n_max=3)
machines["RbmSpin 1d Hypercube boson"] = nk.machine.RbmSpin(hilbert=hi, alpha=1)

machines["RbmSpinSymm 1d Hypercube boson"] = nk.machine.RbmSpinSymm(
    hilbert=hi, alpha=2, automorphisms=g
)
machines["RbmMultiVal 1d Hypercube boson"] = nk.machine.RbmMultiVal(
    hilbert=hi, n_hidden=2
)
machines["Jastrow 1d Hypercube boson"] = nk.machine.Jastrow(hilbert=hi)

machines["JastrowSymm 1d Hypercube boson real"] = nk.machine.JastrowSymm(
    hilbert=hi, dtype=float, automorphisms=g
)
# machines["MPS 1d boson"] = nk.machine.MPSPeriodic(hi, bond_dim=4)


np.random.seed(12346)


def same_derivatives(der_log, num_der_log, eps=1.0e-5):
    assert der_log.shape == num_der_log.shape
    # assert np.max(np.real(der_log - num_der_log)) == approx(0.0, rel=eps, abs=eps)

    np.testing.assert_array_almost_equal(der_log.real, num_der_log.real, decimal=4)

    # The imaginary part is a bit more tricky, there might be an arbitrary phase shift
    assert np.max(
        np.abs(np.exp(np.imag(der_log - num_der_log) * 1.0j) - 1.0)
    ) == approx(0.0, rel=eps, abs=eps)


def log_val_f(par, machine, v):
    machine.parameters = machine.numpy_unflatten(par, machine.parameters)
    if v.ndim != 1:
        if v.size != v.shape[1]:
            raise RuntimeError(
                "numerical derivatives can be tested only for non batched inputs"
            )

    return machine.log_val(v.reshape(1, -1))[0]


def log_val_vec_f(par, machine, v, vec):
    machine.parameters = machine.numpy_unflatten(par, machine.parameters)
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


def test_set_get_parameters():
    for name, machine in merge_dicts(machines, dm_machines).items():
        unflatten = machine.numpy_unflatten
        flatten = machine.numpy_flatten

        print("Machine test: %s" % name)
        assert machine.n_par > 0
        npar = machine.n_par
        machine.init_random_parameters()
        randpars = flatten(machine.parameters)

        if machine.has_complex_parameters:
            assert np.array_equal(flatten(machine.parameters), randpars)
            assert not all(randpars.real == 0)
            assert not all(randpars.imag == 0)
        else:
            assert np.array_equal(flatten(machine.parameters).real, randpars.real)
            assert not all(randpars.real == 0)

        machine.parameters = unflatten(np.zeros(npar), machine.parameters)
        assert np.count_nonzero(np.abs(flatten(machine.parameters))) == 0


def test_save_load_parameters(tmpdir):
    for name, machine in merge_dicts(machines, dm_machines).items():
        print("Machine test: %s" % name)
        assert machine.n_par > 0
        n_par = machine.n_par

        unflatten = machine.numpy_unflatten
        flatten = machine.numpy_flatten

        machine.init_random_parameters()
        randpars = flatten(machine.parameters)

        machine.parameters = unflatten(randpars, machine.parameters)
        fn = tmpdir.mkdir("datawf").join("test.wf")

        filename = os.path.join(fn.dirname, fn.basename)

        machine.save(filename)
        machine.parameters = unflatten(np.zeros(n_par), machine.parameters)
        machine.load(filename)

        os.remove(filename)
        os.rmdir(fn.dirname)
        if machine.has_complex_parameters:
            assert np.array_equal(flatten(machine.parameters), randpars)
        else:
            assert np.array_equal(flatten(machine.parameters).real, randpars.real)


def test_log_derivative():
    np.random.seed(12345)

    for name, machine in merge_dicts(machines, dm_machines).items():

        print("Machine test: %s" % name)

        npar = machine.n_par

        # random visibile state
        hi = machine.hilbert
        assert hi.size > 0

        v = np.zeros(machine.input_size)

        flatten = machine.numpy_flatten

        for i in range(100):
            if name in dm_machines:
                hi.random_state(out=v[: hi.size])
                hi.random_state(out=v[hi.size :])
            else:
                hi.random_state(out=v)

            machine.init_random_parameters(seed=i)
            randpars = flatten(machine.parameters)

            der_log = flatten(machine.der_log(v.reshape((1, -1))))

            assert der_log.shape == (machine.n_par,)

            if "Jastrow" in name:
                assert np.max(np.imag(der_log)) == approx(0.0)

            num_der_log = central_diff_grad(log_val_f, randpars, 1.0e-9, machine, v)

            same_derivatives(der_log, num_der_log)
            # print(np.linalg.norm(der_log - num_der_log))
            # Check if machine is correctly set to be holomorphic
            # The check is done only on smaller subset of parameters, for speed


def test_vector_jacobian():
    for name, machine in merge_dicts(machines, dm_machines).items():
        print("Machine test: %s" % name)

        npar = machine.n_par

        # random visibile state
        hi = machine.hilbert
        if name in dm_machines:
            hi = nk.hilbert.DoubledHilbert(hi)
        assert hi.size > 0

        batch_size = 100
        v = np.zeros((batch_size, machine.input_size))

        flatten = machine.numpy_flatten

        for i in range(batch_size):
            hi.random_state(out=v[i])

        machine.init_random_parameters(seed=1234, sigma=0.1)
        randpars = flatten(machine.parameters)

        vec = np.random.uniform(size=batch_size) + 1.0j * np.random.uniform(
            size=batch_size
        ) / float(batch_size)

        vjp = machine.vector_jacobian_prod(v, vec)
        vjp = flatten(vjp)

        num_der_log = central_diff_grad(
            log_val_vec_f, randpars, 1.0e-6, machine, v, vec
        )

        same_derivatives(vjp, num_der_log)


def test_input_size():
    for name, machine in machines.items():
        print("Machine test: %s" % name)
        hi = machine.hilbert

        assert machine.input_size == hi.size

    for name, machine in dm_machines.items():
        print("Machine test: %s" % name)
        hi = machine.hilbert

        assert machine.input_size == 2 * hi.size


def test_to_array():
    for name, machine in merge_dicts(machines, dm_machines).items():
        print("Machine test: %s" % name)

        npar = machine.n_par

        randpars = 0.5 * (np.random.randn(npar) + 1.0j * np.random.randn(npar))
        if "Torch" in name:
            randpars = randpars.real
        machine.parameters = machine.numpy_unflatten(randpars, machine.parameters)

        hi = machine.hilbert
        if name in dm_machines:
            hi = nk.hilbert.DoubledHilbert(hi)

        all_psis = machine.to_array(normalize=False)
        # test shape
        assert all_psis.shape[0] == hi.n_states
        assert len(all_psis.shape) == 1

        log_vals = machine.log_val(hi.all_states())
        logmax = log_vals.real.max()

        norm = (np.abs(np.exp(log_vals - logmax)) ** 2).sum()

        # Tests broken for dm machines because of norm. should split
        if name in dm_machines:
            continue

        # test random values
        for i in range(100):
            rstate = np.zeros(hi.size)
            local_states = hi.local_states
            hi.random_state(out=rstate)

            number = hi.states_to_numbers(rstate)

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
            hi.random_state(out=rstate)

            number = hi.states_to_numbers(rstate)

            assert np.abs(
                np.exp(machine.log_val(rstate.reshape(1, -1)) - logmax) / np.sqrt(norm)
                - all_psis_normalized[number]
            ) == approx(0.0)
