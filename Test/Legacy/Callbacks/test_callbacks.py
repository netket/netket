import netket.legacy as nk
import time

SEED = 3141592
L = 8


def _vmc(n_iter=20):
    nk.random.seed(SEED)
    hi = nk.hilbert.Spin(s=0.5) ** L

    ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(sigma=0.01, seed=SEED)

    ha = nk.operator.Ising(hi, nk.graph.Hypercube(length=L, n_dim=1), h=1.0)
    sa = nk.sampler.MetropolisLocal(machine=ma)

    op = nk.optimizer.Sgd(ma, learning_rate=0.1)

    return nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=500)

    st = time.time()
    vmc.run(n_iter, callback=callbacks)
    runtime = time.time() - st
    return vmc.step_count, runtime


def test_earlystopping_with_patience():
    patience = 10
    es = nk.callbacks.EarlyStopping(patience=patience)
    es._best_val = -1e6
    vmc = _vmc()

    vmc.run(20, callback=es)

    assert vmc.step_count == patience


def test_earlystopping_with_baseline():
    baseline = -10
    es = nk.callbacks.EarlyStopping(baseline=baseline)
    vmc = _vmc()

    vmc.run(20, callback=es)
    # We should actually assert something..


def test_timeout():
    timeout = 5
    tout = nk.callbacks.Timeout(timeout=timeout)
    vmc = _vmc()

    # warmup the jit
    vmc.run(1)

    st = time.time()
    vmc.run(20000, callback=tout)
    runtime = time.time() - st

    # There is a lag in the first iteration of about 3 seconds
    # But the timeout works!
    assert abs(timeout - runtime) < 3
