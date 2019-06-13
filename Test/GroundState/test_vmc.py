from pytest import approx, raises
import numpy as np

import netket as nk
import netket.variational as vmc

SEED = 214748364


def _setup_vmc():
    L = 4
    g = nk.graph.Hypercube(length=L, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, graph=g)

    ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(seed=SEED, sigma=0.01)

    ha = nk.operator.Ising(hi, h=1.0)
    sa = nk.sampler.ExactSampler(machine=ma)
    sa.seed(SEED)
    op = nk.optimizer.Sgd(learning_rate=0.1)

    # Add custom observable
    X = [[0, 1], [1, 0]]
    sx = nk.operator.LocalOperator(hi, [X] * L, [[i] for i in range(8)])

    driver = nk.variational.Vmc(ha, sa, op, 1000)

    return ha, sx, ma, sa, driver


def test_vmc_functions():
    ha, sx, ma, sampler, driver = _setup_vmc()

    driver.advance(200)

    state = ma.to_array()

    exact_dist = np.abs(state) ** 2

    for op, name, tol in (ha, "ha", 1e-6), (sx, "sx", 1e-2):
        print("Testing expectation of op={}".format(name))

        exact_locs = [vmc.local_value(op, ma, v) for v in ma.hilbert.states()]
        exact_ex = np.sum(exact_dist * exact_locs).real

        data = vmc.compute_samples(sampler, nsamples=10000, ndiscard=1000)

        ex, lv = vmc.expectation(data, ma, op, return_locvals=True)
        assert ex["Mean"] == approx(np.mean(lv).real, rel=tol)
        assert ex["Mean"] == approx(exact_ex, rel=tol)

    var = vmc.variance(data, ma, ha)
    assert var["Mean"] == approx(0.0, abs=1e-7)

    grad = vmc.gradient(data, ma, ha)
    assert grad.shape == (ma.n_par,)
    assert np.mean(np.abs(grad) ** 2) == approx(0.0, abs=1e-9)

    data_without_logderivs = vmc.compute_samples(
        sampler, nsamples=10000, compute_logderivs=False
    )
    with raises(
        RuntimeError,
        match="vmc::Result does not contain log-derivatives, "
        "which are required to compute gradients.",
    ):
        vmc.gradient(data_without_logderivs, ma, ha)
