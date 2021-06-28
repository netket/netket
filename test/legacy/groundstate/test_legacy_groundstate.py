import json
from pytest import approx
import netket.legacy as nk
import numpy as np
import shutil
import tempfile

import pytest

pytestmark = pytest.mark.legacy


SEED = 3141592


def _setup_vmc(
    n_samples=200, diag_shift=0, use_iterative=False, lsq_solver=None, **kwargs
):
    L = 8
    nk.random.seed(SEED)
    hi = nk.hilbert.Spin(s=0.5) ** L

    ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(sigma=0.01, seed=SEED)

    ha = nk.operator.Ising(hi, nk.graph.Hypercube(length=L, n_dim=1), h=1.0)
    sa = nk.sampler.MetropolisLocal(machine=ma)

    op = nk.optimizer.Sgd(ma, learning_rate=0.1)

    vmc = nk.Vmc(ha, sa, op, n_samples=n_samples, **kwargs)

    # Add custom observable
    X = [[0, 1], [1, 0]]
    sx = nk.operator.LocalOperator(hi, [X] * 8, [[i] for i in range(8)])

    return ma, vmc, sx


def test_vmc_advance():
    ma1, vmc1, _ = _setup_vmc(n_samples=500, diag_shift=0.01)
    for i in range(10):
        vmc1.advance()

    ma2, vmc2, _ = _setup_vmc(n_samples=500, diag_shift=0.01)
    for step in vmc2.iter(10):
        pass

    assert (ma1.parameters == ma2.parameters).all()


def test_vmc_advance_iterative():
    ma1, vmc1, _ = _setup_vmc(n_samples=500, diag_shift=0.01, use_iterative=True)
    for i in range(10):
        vmc1.advance()

    ma2, vmc2, _ = _setup_vmc(n_samples=500, diag_shift=0.01, use_iterative=True)
    for step in vmc2.iter(10):
        pass

    assert (ma1.parameters == ma2.parameters).all()


def test_vmc_iterator():
    ma, vmc, sx = _setup_vmc(n_samples=500, diag_shift=0.01)
    operators = {"Energy": vmc._ham, "SigmaX": sx}

    count = 0
    last_obs = None
    for i, step in enumerate(vmc.iter(300)):
        count += 1
        assert step == i
        obs = vmc.estimate(operators)
        for name in "Energy", "SigmaX":
            assert name in obs
            e = obs[name]
            assert (
                hasattr(e, "mean")
                and hasattr(e, "error_of_mean")
                and hasattr(e, "variance")
                and hasattr(e, "tau_corr")
                and hasattr(e, "R_hat")
            )
        last_obs = obs

    assert count == 300
    assert last_obs["Energy"].mean == approx(-10.25, abs=0.2)


def test_vmc_iterator_iterative():
    ma, vmc, sx = _setup_vmc(n_samples=500, diag_shift=0.01, use_iterative=True)
    operators = {"Energy": vmc._ham, "SigmaX": sx}

    count = 0
    last_obs = None
    for i, step in enumerate(vmc.iter(300)):
        count += 1
        assert step == i
        obs = vmc.estimate(operators)
        # TODO: Choose which version we want
        # for name in "Energy", "EnergyVariance", "SigmaX":
        #     assert name in obs
        #     e = obs[name]
        #     assert "Mean" in e and "Sigma" in e and "Taucorr" in e
        for name in "Energy", "SigmaX":
            assert name in obs
            e = obs[name]
            assert hasattr(e, "mean") and hasattr(e, "variance") and hasattr(e, "R_hat")
        last_obs = obs

    assert count == 300
    assert last_obs["Energy"].mean == approx(-10.25, abs=0.2)


def test_vmc_run():
    ma, vmc, sx = _setup_vmc(n_samples=500, diag_shift=0.01)
    obs = {"SigmaX": sx}

    tempdir = tempfile.mkdtemp()
    print("Writing test output files to: {}".format(tempdir))
    prefix = tempdir + "/vmc_test"
    vmc.run(300, prefix, obs)

    with open(prefix + ".log") as logfile:
        log = json.load(logfile)

    shutil.rmtree(tempdir)

    assert "Output" in log
    output = log["Output"]
    assert len(output) == 300

    for i, obs in enumerate(output):
        step = obs["Iteration"]
        assert step == i
        for name in "Energy", "SigmaX":
            assert name in obs
            e = obs[name]
            assert "Mean" in e
            assert "Sigma" in e
            assert "TauCorr" in e
        last_obs = obs

    assert last_obs["Energy"]["Mean"] == approx(-10.25, abs=0.2)


def test_imag_time_propagation():
    L = 8
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, N=L)
    ha = nk.operator.Ising(graph=g, h=0.0, hilbert=hi)

    psi0 = np.random.rand(hi.n_states)
    driver = nk.exact.PyExactTimePropagation(
        ha,
        t0=0,
        dt=0.1,
        initial_state=psi0,
        propagation_type="imaginary",
        solver_kwargs={"atol": 1e-10, "rtol": 1e-10},
    )

    driver.advance(1000)
    assert driver.estimate(ha).mean.real == approx(-8.0)
