from pytest import approx, raises
import numpy as np

import netket.legacy as nk
import netket.legacy.variational as vmc

from io import StringIO
from contextlib import redirect_stderr
import tempfile
import re

import pytest

pytestmark = pytest.mark.legacy

SEED = 214748364
nk.random.seed(SEED)


def _setup_vmc(lsq_solver=None):
    L = 4
    g = nk.graph.Hypercube(length=L, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(sigma=0.01)

    ha = nk.operator.Ising(hi, graph=g, h=1.0)
    sa = nk.sampler.ExactSampler(machine=ma, sample_size=16)
    op = nk.optimizer.Sgd(ma, learning_rate=0.05)

    # Add custom observable
    X = [[0, 1], [1, 0]]
    sx = nk.operator.LocalOperator(hi, [X] * L, [[i] for i in range(8)])

    sr = nk.optimizer.SR(ma, use_iterative=False, lsq_solver=lsq_solver)
    driver = nk.Vmc(ha, sa, op, 1000, sr=sr)

    return ha, sx, ma, sa, driver


def test_before_first_step():
    ha, *_, driver = _setup_vmc()
    with raises(RuntimeError):
        doesnotwork = driver.estimate(ha)
    driver.advance(1)
    driver.estimate(ha)


def test_vmc_functions():
    ha, sx, ma, sampler, driver = _setup_vmc()
    driver.advance(500)

    state = ma.to_array()

    exact_dist = np.abs(state) ** 2

    n_samples = 16000

    for op, name in (ha, "ha"), (sx, "sx"):
        print("Testing expectation of op={}".format(name))

        states = np.array(list(ma.hilbert.states()))
        exact_locs = nk.operator.local_values(op, ma, states)
        exact_ex = np.sum(exact_dist * exact_locs).real

        for _ in sampler.samples(10):
            pass
        samples = sampler.generate_samples(n_samples).reshape((-1, ma.hilbert.size))

        der_logs = ma.der_log(samples)
        der_logs -= nk.stats.mean(der_logs)

        local_values = nk.operator.local_values(op, ma, samples)

        mean = local_values.mean()
        var = local_values.var()
        print(mean, var)

        # 5-sigma test for expectation values
        tol = np.sqrt(var / float(local_values.size)) * 5
        assert mean.real == approx(exact_ex.real, abs=tol)

        stats = vmc.estimate_expectations(
            op, sampler, n_samples=n_samples, n_discard=10
        )

        assert stats.mean.real == approx(mean.real, rel=tol)
        assert stats.mean.real == approx(exact_ex.real, abs=tol)

    local_values = nk.operator.local_values(ha, ma, samples)
    # grad = nk.stats.covariance_sv(local_values, der_logs)
    # assert grad.shape == (ma.n_par,)
    # assert np.mean(np.abs(grad) ** 2) == approx(0.0, abs=1e-8)

    _, grads = vmc.estimate_expectations(
        ha, sampler, n_samples, 10, compute_gradients=True
    )

    assert grads.shape == (ma.n_par,)
    assert np.mean(np.abs(grads) ** 2) == approx(0.0, abs=1e-8)


def test_raise_n_iter():
    ha, sx, ma, sampler, driver = _setup_vmc()
    with raises(
        ValueError,
    ):
        driver.run("prova", 12)


def test_vmc_use_cholesky_compatibility():
    ha, _, ma, sampler, _ = _setup_vmc(lsq_solver="Cholesky")


def test_vmc_progress_bar():
    ha, sx, ma, sampler, driver = _setup_vmc()
    tempdir = tempfile.mkdtemp()
    prefix = tempdir + "/vmc_progressbar_test"

    f = StringIO()
    with redirect_stderr(f):
        driver.run(5, prefix)
    pbar = f.getvalue().split("\r")[-1]
    assert re.match(r"100%\|#*\| (\d+)/\1", pbar)
    assert re.search(r"Energy=[-+]?[0-9]*\.?[0-9]*", pbar)
    assert re.search(r"σ²=[-+]?[0-9]*\.?[0-9]*", pbar)

    f = StringIO()
    with redirect_stderr(f):
        driver.run(5, prefix, show_progress=None)
    pbar = f.getvalue()
    assert not len(pbar)


def _energy(par, machine, H):
    machine.parameters = np.copy(par)
    psi = machine.to_array()
    return np.real(psi.conj() @ H @ psi)


def central_diff_grad(func, x, eps, *args):
    grad = np.zeros(len(x), dtype=complex)
    epsd = np.zeros(len(x), dtype=complex)
    epsd[0] = eps
    for i in range(len(x)):
        assert not np.any(np.isnan(x + epsd))
        grad_r = 0.5 * (func(x + epsd, *args) - func(x - epsd, *args))
        grad_i = 0.5 * (func(x + 1j * epsd, *args) - func(x - 1j * epsd, *args))
        grad[i] = 0.5 * grad_r + 0.5j * grad_i
        assert not np.isnan(grad[i])
        grad[i] /= eps
        epsd = np.roll(epsd, 1)
    return grad


def same_derivatives(der_log, num_der_log, abs_eps=1.0e-6, rel_eps=1.0e-6):
    assert np.max(np.real(der_log - num_der_log)) == approx(
        0.0, rel=rel_eps, abs=abs_eps
    )
    # The imaginary part is a bit more tricky, there might be an arbitrary phase shift
    assert np.max(
        np.abs(np.exp(np.imag(der_log - num_der_log) * 1.0j) - 1.0)
    ) == approx(0.0, rel=rel_eps, abs=abs_eps)


def test_vmc_gradient():
    ha, sx, ma, sampler, driver = _setup_vmc()
    pars = np.copy(ma.parameters)
    driver._sr = None
    grad_exact = central_diff_grad(_energy, pars, 1.0e-5, ma, ha.to_sparse())

    driver.n_samples = 1e6
    driver.n_discard = 1e3
    driver.machine.parameters = np.copy(pars)
    grad_approx = driver._forward_and_backward()

    err = 6 / np.sqrt(driver.n_samples)
    same_derivatives(grad_approx, grad_exact, abs_eps=err, rel_eps=1.0e-3)
