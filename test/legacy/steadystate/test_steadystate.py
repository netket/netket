import pytest

import json
from pytest import approx
import netket.legacy as nk
import numpy as np
import shutil
import tempfile

pytestmark = pytest.mark.legacy

SEED = 3141592
L = 4

sx = [[0, 1], [1, 0]]
sy = [[0, -1j], [1j, 0]]
sz = [[1, 0], [0, -1]]

sigmam = [[0, 0], [1, 0]]


def _setup_system():
    hi = nk.hilbert.Spin(s=0.5) ** L

    ha = nk.operator.LocalOperator(hi)
    j_ops = []
    for i in range(L):
        ha += (0.3 / 2.0) * nk.operator.LocalOperator(hi, sx, [i])
        ha += (2.0 / 4.0) * nk.operator.LocalOperator(
            hi, np.kron(sz, sz), [i, (i + 1) % L]
        )
        j_ops.append(nk.operator.LocalOperator(hi, sigmam, [i]))

    # Create the liouvillian
    lind = nk.operator.LocalLiouvillian(ha, j_ops)
    return hi, lind


def _setup_ss(**kwargs):
    nk.random.seed(SEED)
    np.random.seed(SEED)

    hi, lind = _setup_system()

    ma = nk.machine.density_matrix.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(sigma=0.01, seed=SEED)

    sa = nk.sampler.MetropolisLocal(machine=ma)
    sa_obs = nk.sampler.MetropolisLocal(machine=ma.diagonal())

    op = nk.optimizer.Sgd(ma, learning_rate=0.1)

    if "sr" in kwargs:
        sr = nk.optimizer.SR(ma, **kwargs["sr"])
        kwargs["sr"] = sr

    ss = nk.SteadyState(
        lindblad=lind, sampler=sa, optimizer=op, sampler_obs=sa_obs, **kwargs
    )

    return ma, ss


def _setup_obs():
    hi = nk.hilbert.Spin(s=0.5) ** L

    obs_sx = nk.operator.LocalOperator(hi)
    for i in range(L):
        obs_sx += nk.operator.LocalOperator(hi, sx, [i])

    obs = {"SigmaX": obs_sx}
    return obs


def test_ss_advance():
    ma1, vmc1 = _setup_ss(n_samples=500, n_samples_obs=250)
    for i in range(10):
        vmc1.advance()

    ma2, vmc2 = _setup_ss(n_samples=500, n_samples_obs=250)
    for step in vmc2.iter(10):
        pass

    assert (ma1.parameters == ma2.parameters).all()


def test_ss_advance_sr():
    sr = {"diag_shift": 0.01, "use_iterative": False}

    ma1, vmc1 = _setup_ss(n_samples=500, n_samples_obs=250, sr=sr)
    for i in range(10):
        vmc1.advance()

    sr = {"diag_shift": 0.01, "use_iterative": False}
    ma2, vmc2 = _setup_ss(n_samples=500, n_samples_obs=250, sr=sr)
    for step in vmc2.iter(10):
        pass

    assert (ma1.parameters == ma2.parameters).all()


def test_ss_advance_sr_iterative():
    sr = {"diag_shift": 0.01, "use_iterative": True}
    ma1, vmc1 = _setup_ss(n_samples=500, n_samples_obs=250, sr=sr)
    for i in range(10):
        vmc1.advance()

    sr = {"diag_shift": 0.01, "use_iterative": True}
    ma2, vmc2 = _setup_ss(n_samples=500, n_samples_obs=250, sr=sr)
    for step in vmc2.iter(10):
        pass

    assert (ma1.parameters == ma2.parameters).all()


def test_ss_iterator():
    ma, vmc = _setup_ss(n_samples=800, n_samples_obs=250)
    obs_op = _setup_obs()

    N_iters = 160
    count = 0
    losses = []
    for i, step in enumerate(vmc.iter(N_iters)):
        count += 1
        assert step == i
        obs = vmc.estimate(obs_op)
        obs["LdagL"] = vmc.ldagl
        for name in "LdagL", "SigmaX":
            assert name in obs
            e = obs[name]
            assert (
                hasattr(e, "mean")
                and hasattr(e, "error_of_mean")
                and hasattr(e, "variance")
                and hasattr(e, "tau_corr")
                and hasattr(e, "R_hat")
            )
        losses.append(vmc.ldagl["mean"])

    assert count == N_iters
    assert np.mean(losses[-10:]) == approx(0.0, abs=0.003)


def test_ss_iterator_sr():
    sr = {"diag_shift": 0.01, "use_iterative": False}
    ma, vmc = _setup_ss(n_samples=800, n_samples_obs=250, sr=sr)
    obs_op = _setup_obs()

    N_iters = 160
    count = 0
    losses = []
    for i, step in enumerate(vmc.iter(N_iters)):
        count += 1
        assert step == i
        obs = vmc.estimate(obs_op)
        obs["LdagL"] = vmc.ldagl
        # TODO: Choose which version we want
        # for name in "Energy", "EnergyVariance", "SigmaX":
        #     assert name in obs
        #     e = obs[name]
        #     assert "Mean" in e and "Sigma" in e and "Taucorr" in e
        for name in "LdagL", "SigmaX":
            assert name in obs
            e = obs[name]
            assert hasattr(e, "mean") and hasattr(e, "variance") and hasattr(e, "R_hat")
        losses.append(vmc.ldagl["mean"])

    assert count == N_iters
    # Should readd, but pure python machines are not good enough to converge
    # requires jax. split into different file?
    # assert np.mean(losses[-10:]) == approx(0.0, abs=0.003)


def test_ss_run():
    ma, vmc = _setup_ss(n_samples=1000, n_samples_obs=250)
    obs_op = _setup_obs()

    N_iters = 200

    tempdir = tempfile.mkdtemp()
    print("Writing test output files to: {}".format(tempdir))
    prefix = tempdir + "/ss_test"
    vmc.run(N_iters, prefix, obs=obs_op)

    with open(prefix + ".log") as logfile:
        log = json.load(logfile)

    shutil.rmtree(tempdir)

    assert "Output" in log
    output = log["Output"]
    assert len(output) == N_iters

    losses = []
    for i, obs in enumerate(output):
        step = obs["Iteration"]
        assert step == i
        for name in "LdagL", "SigmaX":
            assert name in obs
            e = obs[name]
            assert "Mean" in e
            assert "Sigma" in e
            assert "TauCorr" in e
        losses.append(vmc.ldagl["Mean"])

    # Should readd, but pure python machines are not good enough to converge
    # requires jax. split into different file?
    # assert np.mean(losses[-10:]) == approx(0.0, abs=0.003)


def _ldagl(par, machine, L):
    machine.parameters = np.copy(par)
    rho = machine.to_array()
    return np.sum(np.abs(L @ rho) ** 2)


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


def same_derivatives(der_log, num_der_log, eps=1.0e-6):
    assert np.max(np.real(der_log - num_der_log)) == approx(0.0, rel=eps, abs=eps)
    # The imaginary part is a bit more tricky, there might be an arbitrary phase shift
    assert np.max(
        np.abs(np.exp(np.imag(der_log - num_der_log) * 1.0j) - 1.0)
    ) == approx(0.0, rel=eps, abs=eps)


# disable test because to make it work we need MUCH more samples than travis
# can handle
def _test_ldagl_gradient():
    ma, driver = _setup_ss(n_samples=500, n_samples_obs=250)
    lind = driver._lind
    pars = np.copy(ma.parameters)

    driver._sr = None
    grad_exact = central_diff_grad(_ldagl, pars, 1.0e-5, ma, lind.to_sparse())

    driver.n_samples = 2e5
    driver.n_discard = 5e2
    driver.machine.parameters = np.copy(pars)
    grad_approx = driver._forward_and_backward()

    err = 5 / np.sqrt(driver.n_samples)
    same_derivatives(np.real(grad_approx), np.real(grad_exact), eps=err)
