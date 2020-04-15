from pytest import approx, raises
import numpy as np

import netket as nk
import netket.variational as vmc

from io import StringIO
from contextlib import redirect_stderr
import tempfile
import re

SEED = 214748364
nk.utils.seed(SEED)


def _setup_vmc():
    L = 4
    g = nk.graph.Hypercube(length=L, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, graph=g)

    ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(sigma=0.01)

    ha = nk.operator.Ising(hi, h=1.0)
    sa = nk.sampler.ExactSampler(machine=ma, sample_size=16)
    op = nk.optimizer.Sgd(learning_rate=0.05)

    # Add custom observable
    X = [[0, 1], [1, 0]]
    sx = nk.operator.LocalOperator(hi, [X] * L, [[i] for i in range(8)])

    driver = nk.variational.Vmc(ha, sa, op, 1000)

    return ha, sx, ma, sa, driver


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
        samples = np.array(list(sampler.samples(n_samples)))
        der_logs = ma.der_log(samples)
        nk.stats.subtract_mean(der_logs)

        local_values = nk.operator.local_values(op, ma, samples)

        ex = nk.stats.statistics(local_values)

        # 5-sigma test for expectation values
        tol = ex.error_of_mean * 5
        assert ex.mean.real == approx(np.mean(local_values).real, abs=tol)
        assert ex.mean.real == approx(exact_ex.real, abs=tol)

        stats = vmc.estimate_expectations(
            op, sampler, n_samples=n_samples, n_discard=10
        )

        assert stats.mean.real == approx(np.mean(local_values).real, rel=tol)
        assert stats.mean.real == approx(exact_ex.real, abs=tol)

    local_values = nk.operator.local_values(ha, ma, samples)
    grad = nk.stats.covariance_sv(local_values, der_logs)
    assert grad.shape == (ma.n_par,)
    assert np.mean(np.abs(grad) ** 2) == approx(0.0, abs=1e-8)

    _, grads = vmc.estimate_expectations(
        ha, sampler, n_samples, 10, compute_gradients=True
    )

    assert grads.shape == (ma.n_par,)
    assert np.mean(np.abs(grads) ** 2) == approx(0.0, abs=1e-8)


def test_vmc_use_cholesky_compatibility():
    ha, _, ma, sampler, _ = _setup_vmc()

    op = nk.optimizer.Sgd(learning_rate=0.1)
    with raises(
        ValueError,
        match="Inconsistent options specified: `use_cholesky && sr_lsq_solver != 'LLT'`.",
    ):
        vmc = nk.variational.Vmc(
            ha, sampler, op, 1000, use_cholesky=True, sr_lsq_solver="BDCSVD"
        )


def test_vmc_progress_bar():
    ha, sx, ma, sampler, driver = _setup_vmc()
    driver.add_observable(sx, "sx")
    tempdir = tempfile.mkdtemp()
    prefix = tempdir + "/vmc_progressbar_test"

    f = StringIO()
    with redirect_stderr(f):
        driver.run(prefix, 5)
    pbar = f.getvalue().split("\r")[-1]
    assert re.match(r"100%\|#*\| (\d+)/\1", pbar)
    assert re.search(r"Energy=\([-+]?[0-9]*\.?[0-9]*", pbar)
    assert re.search(r"var=[-+]?[0-9]*\.?[0-9]*", pbar)

    f = StringIO()
    with redirect_stderr(f):
        driver.run(prefix, 5, show_progress=None)
    pbar = f.getvalue()
    assert not len(pbar)
