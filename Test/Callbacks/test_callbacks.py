import netket as nk
import time

SEED = 3141592


def _run_vmc(callbacks, n_iter=20):
    nk.random.seed(SEED)
    g = nk.graph.Hypercube(length=8, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, graph=g)

    ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(sigma=0.01, seed=SEED)

    ha = nk.operator.Ising(hi, h=1.0)
    sa = nk.sampler.MetropolisLocal(machine=ma)

    op = nk.optimizer.Sgd(ma, learning_rate=0.1)

    vmc = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=500)
    st = time.time()
    vmc.run(n_iter, callbacks=callbacks)
    runtime = time.time() - st
    return vmc.step_count, runtime


def test_earlystopping_with_patience():
    patience = 10
    es = nk.callbacks.EarlyStopping(patience=patience)
    es._best_val = -1e6
    step_value = _run_vmc([es])
    assert step_value, runtime == patience


def test_earlystopping_with_baseline():
    baseline = -10
    es = nk.callbacks.EarlyStopping(baseline=baseline)
    _step_value, runtime = _run_vmc([es])


def test_timeout():
    timeout = 5
    tout = nk.callbacks.Timeout(timeout=timeout)
    step_value, runtime = _run_vmc([tout], 300)

    # There is a lag in the first iteration of about 3 seconds
    # But the timeout works!
    assert abs(timeout - runtime) < 3
