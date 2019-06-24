import json
from pytest import approx
import netket as nk
import numpy as np
import shutil
import tempfile


SEED = 3141592


def _setup_vmc():
    g = nk.graph.Hypercube(length=8, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, graph=g)

    ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(seed=SEED, sigma=0.01)

    ha = nk.operator.Ising(hi, h=1.0)
    sa = nk.sampler.MetropolisLocal(machine=ma)
    sa.seed(SEED)
    op = nk.optimizer.Sgd(learning_rate=0.1)

    vmc = nk.variational.Vmc(
        hamiltonian=ha, sampler=sa, optimizer=op, n_samples=500, diag_shift=0.01
    )

    # Add custom observable
    X = [[0, 1], [1, 0]]
    sx = nk.operator.LocalOperator(hi, [X] * 8, [[i] for i in range(8)])
    vmc.add_observable(sx, "SigmaX")

    return ma, vmc


def test_vmc_advance():
    ma1, vmc1 = _setup_vmc()
    ma2, vmc2 = _setup_vmc()

    for i in range(10):
        vmc1.advance()

    for step in vmc2.iter(10):
        pass

    assert (ma1.parameters == ma2.parameters).all()


def test_vmc_iterator():
    ma, vmc = _setup_vmc()

    count = 0
    last_obs = None
    for i, step in enumerate(vmc.iter(300)):
        count += 1
        assert step == i
        obs = vmc.get_observable_stats()
        for name in "Energy", "EnergyVariance", "SigmaX":
            assert name in obs
            e = obs[name]
            assert "Mean" in e and "Sigma" in e and "Taucorr" in e
        last_obs = obs

    assert count == 300
    assert last_obs["Energy"]["Mean"] == approx(-10.25, abs=0.2)


def test_vmc_run():
    ma, vmc = _setup_vmc()

    tempdir = tempfile.mkdtemp()
    print("Writing test output files to: {}".format(tempdir))
    prefix = tempdir + "/vmc_test"
    vmc.run(prefix, 300)

    with open(prefix + ".log") as logfile:
        log = json.load(logfile)

    shutil.rmtree(tempdir)

    assert "Output" in log
    output = log["Output"]
    assert len(output) == 300

    for i, obs in enumerate(output):
        step = obs["Iteration"]
        assert step == i
        for name in "Energy", "EnergyVariance", "SigmaX":
            assert name in obs
            e = obs[name]
            assert "Mean" in e and "Sigma" in e and "Taucorr" in e
        last_obs = obs

    assert last_obs["Energy"]["Mean"] == approx(-10.25, abs=0.2)


def test_imag_time_propagation():
    g = nk.graph.Hypercube(length=8, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    ha = nk.operator.Ising(h=0.0, hilbert=hi)

    stepper = nk.dynamics.timestepper(hi.n_states, rel_tol=1e-10, abs_tol=1e-10)
    psi0 = np.random.rand(hi.n_states)
    driver = nk.exact.ExactTimePropagation(
        ha, stepper, t0=0, initial_state=psi0, propagation_type="imaginary"
    )

    for step in driver.iter(dt=0.1, n_iter=1000):
        pass

    assert driver.get_observable_stats()["Energy"]["Mean"] == approx(-8.0)


def test_ed():
    first_n = 3
    g = nk.graph.Hypercube(length=8, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    ha = nk.operator.Ising(h=1.0, hilbert=hi)

    # Test Lanczos ED with eigenvectors
    res = nk.exact.lanczos_ed(ha, first_n=first_n, compute_eigenvectors=True)
    assert len(res.eigenvalues) == first_n
    assert len(res.eigenvectors) == first_n
    gse = res.mean(ha, 0)
    fse = res.mean(ha, 1)
    assert gse == approx(res.eigenvalues[0], rel=1e-12, abs=1e-12)
    assert fse == approx(res.eigenvalues[1], rel=1e-12, abs=1e-12)

    # Test Lanczos ED without eigenvectors
    res = nk.exact.lanczos_ed(ha, first_n=first_n, compute_eigenvectors=False)
    assert len(res.eigenvalues) == first_n
    assert len(res.eigenvectors) == 0

    # Test Full ED with eigenvectors
    res = nk.exact.full_ed(ha, first_n=first_n, compute_eigenvectors=True)
    assert len(res.eigenvalues) == first_n
    assert len(res.eigenvectors) == first_n
    gse = res.mean(ha, 0)
    fse = res.mean(ha, 1)
    assert gse == approx(res.eigenvalues[0], rel=1e-12, abs=1e-12)
    assert fse == approx(res.eigenvalues[1], rel=1e-12, abs=1e-12)

    # Test Full ED without eigenvectors
    res = nk.exact.full_ed(ha, first_n=first_n, compute_eigenvectors=False)
    assert len(res.eigenvalues) == first_n
    assert len(res.eigenvectors) == 0
