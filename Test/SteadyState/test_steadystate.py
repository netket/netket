import json
from pytest import approx
import netket as nk
import numpy as np
import shutil
import tempfile


SEED = 3141592
L = 4

sx = [[0, 1], [1, 0]]
sy = [[0, -1j], [1j, 0]]
sz = [[1, 0], [0, -1]]

sigmam = [[0, 0], [1, 0]]


def _setup_ss(**kwargs):
    nk.utils.seed(SEED)
    g = nk.graph.Hypercube(length=L, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, graph=g)

    ma = nk.machine.NdmSpinPhase(hilbert=hi, alpha=1, beta=1)
    ma.init_random_parameters(sigma=0.01)

    ha = nk.operator.LocalOperator(hi)
    obs_sx = nk.operator.LocalOperator(hi)
    j_ops = []
    for i in range(L):
        ha += (0.3 / 2.0) * nk.operator.LocalOperator(hi, sx, [i])
        ha += (2.0 / 4.0) * nk.operator.LocalOperator(hi, np.kron(sz, sz), [i, (i + 1) % L])
        j_ops.append(nk.operator.LocalOperator(hi, sigmam, [i]))

        obs_sx += nk.operator.LocalOperator(hi, sx, [i])

    #  Create the liouvillian
    lind = nk.operator.LocalLiouvillian(ha, j_ops)

    sa = nk.sampler.MetropolisLocal(machine=ma)
    sa_obs = nk.sampler.MetropolisLocal(machine=ma.diagonal())

    op = nk.optimizer.Sgd(learning_rate=0.1)

    ss = nk.SteadyState(lindblad=lind, sampler=sa, optimizer=op, sampler_obs=sa_obs,
                            **kwargs)

    # Add custom observable
    ss.add_observable(obs_sx, "SigmaX")

    return ma, ss


def test_ss_advance():
    ma1, vmc1 = _setup_ss(n_samples=500, n_samples_obs=250)
    for i in range(10):
        vmc1.advance()

    ma2, vmc2 = _setup_ss(n_samples=500, n_samples_obs=250)
    for step in vmc2.iter(10):
        pass

    assert (ma1.parameters == ma2.parameters).all()


def test_ss_advance_sr():
    sr = nk.optimizer.SR(diag_shift=0.01, use_iterative=False)

    ma1, vmc1 = _setup_ss(n_samples=500, n_samples_obs=250, sr=sr)
    for i in range(10):
        vmc1.advance()

    sr = nk.optimizer.SR(diag_shift=0.01, use_iterative=False)
    ma2, vmc2 = _setup_ss(n_samples=500, n_samples_obs=250, sr=sr)
    for step in vmc2.iter(10):
        pass

    assert (ma1.parameters == ma2.parameters).all()


def test_ss_advance_sr_iterative():
    sr = nk.optimizer.SR(diag_shift=0.01, use_iterative=True)
    ma1, vmc1 = _setup_ss(n_samples=500, n_samples_obs=250, sr=sr)
    for i in range(10):
        vmc1.advance()

    sr = nk.optimizer.SR(diag_shift=0.01, use_iterative=True)
    ma2, vmc2 = _setup_ss(n_samples=500, n_samples_obs=250, sr=sr)
    for step in vmc2.iter(10):
        pass

    assert (ma1.parameters == ma2.parameters).all()


def test_ss_iterator():
    ma, vmc = _setup_ss(n_samples=700, n_samples_obs=250)

    N_iters = 150
    count = 0
    last_obs = None
    for i, step in enumerate(vmc.iter(N_iters)):
        count += 1
        assert step == i
        obs = vmc.get_observable_stats()
        for name in "LdagL", "SigmaX":
            assert name in obs
            e = obs[name]
            assert (
                    hasattr(e, "mean")
                    and hasattr(e, "error_of_mean")
                    and hasattr(e, "variance")
                    and hasattr(e, "tau_corr")
                    and hasattr(e, "R")
            )
        last_obs = obs

    assert count == N_iters
    assert last_obs["LdagL"].mean == approx(0.0, abs=0.001)

def test_ss_iterator_iterative():
    ma, vmc = _setup_ss(n_samples=700, n_samples_obs=250)

    N_iters = 150
    count = 0
    last_obs = None
    for i, step in enumerate(vmc.iter(N_iters)):
        count += 1
        assert step == i
        obs = vmc.get_observable_stats()
        # TODO: Choose which version we want
        # for name in "Energy", "EnergyVariance", "SigmaX":
        #     assert name in obs
        #     e = obs[name]
        #     assert "Mean" in e and "Sigma" in e and "Taucorr" in e
        for name in "LdagL", "SigmaX":
            assert name in obs
            e = obs[name]
            assert hasattr(e, "mean") and hasattr(e, "variance") and hasattr(e, "R")
        last_obs = obs

    assert count == N_iters
    assert last_obs["LdagL"].mean == approx(0.0, abs=0.001)


def test_ss_run():
    ma, vmc = _setup_ss(n_samples=700, n_samples_obs=250)

    N_iters = 150

    tempdir = tempfile.mkdtemp()
    print("Writing test output files to: {}".format(tempdir))
    prefix = tempdir + "/ss_test"
    vmc.run(prefix, N_iters)

    with open(prefix + ".log") as logfile:
        log = json.load(logfile)

    shutil.rmtree(tempdir)

    assert "Output" in log
    output = log["Output"]
    assert len(output) == N_iters

    for i, obs in enumerate(output):
        step = obs["Iteration"]
        assert step == i
        for name in "LdagL", "SigmaX":
            assert name in obs
            e = obs[name]
            assert "Mean" in e
            assert "Sigma" in e
            assert "TauCorr" in e
        last_obs = obs

    assert last_obs["LdagL"]["Mean"] == approx(0.0, abs=0.001)
