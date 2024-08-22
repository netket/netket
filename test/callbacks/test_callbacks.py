import netket as nk
import netket.experimental as nkx
import time
import numpy as np
import flax
import pytest

SEED = 3141592
L = 8


class DummyDriver:
    _loss_name: str = "loss"


class DummyLogEntry:
    def __init__(self, mean):
        self.mean = mean


def _vmc(n_iter=20):
    hi = nk.hilbert.Spin(s=0.5) ** L

    ma = nk.models.RBM(alpha=1)

    ha = nk.operator.IsingJax(hi, nk.graph.Hypercube(length=L, n_dim=1), h=1.0)
    sa = nk.sampler.MetropolisLocal(hi)
    vs = nk.vqs.MCState(sa, ma, n_samples=512, seed=SEED)

    op = nk.optimizer.Sgd(learning_rate=0.1)

    return nk.VMC(hamiltonian=ha, variational_state=vs, optimizer=op)


def _tdvp(n_iter=20):
    hi = nk.hilbert.Spin(s=0.5) ** L

    ma = nk.models.RBM(alpha=1)
    # rescale so that dt=1.0
    ha = 1e-2 * nk.operator.IsingJax(hi, nk.graph.Hypercube(length=L, n_dim=1), h=1.0)
    sa = nk.sampler.MetropolisLocal(hi)
    vs = nk.vqs.MCState(sa, ma, n_samples=512, seed=SEED)

    int = nkx.dynamics.RK4(dt=1.0)
    solv = nk.optimizer.solver.svd(rcond=1e-5)

    return nkx.TDVP(
        operator=ha, variational_state=vs, integrator=int, linear_solver=solv
    )


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


def test_earlystopping_doesnt_get_stuck_with_patience():
    loss_values = [10] + [9] * 12 + [1] * 4
    es = nk.callbacks.EarlyStopping(patience=10)
    driver = DummyDriver()
    for step in range(len(loss_values)):
        print(es)
        if not es(step, {"loss": DummyLogEntry(loss_values[step])}, driver):
            break

    assert step == 12
    assert es._best_iter == 1
    assert es._best_val == 9


def test_earlystopping_doesnt_get_stuck_with_patience_reltol():
    loss_values = np.array([11] + [10] * 12 + [9] * 4, dtype=float)
    loss_values[1:13] = 10.0 - 1e-3 * np.arange(12)
    es = nk.callbacks.EarlyStopping(patience=10, min_reldelta=1.5e-3)
    driver = DummyDriver()
    for step in range(len(loss_values)):
        if not es(step, {"loss": DummyLogEntry(loss_values[step])}, driver):
            break

    assert step == 12
    assert es._best_iter == 1
    assert es._best_val == 10.0

    es = nk.callbacks.EarlyStopping(patience=10, min_reldelta=1e-4)
    driver = DummyDriver()
    for step in range(len(loss_values)):
        if not es(step, {"loss": DummyLogEntry(loss_values[step])}, driver):
            break

    assert step == 16
    assert es._best_iter == 13
    assert es._best_val == 9


@pytest.mark.parametrize("driver", [_tdvp(), _vmc()])
def test_invalid_loss_stopping(driver):
    patience = 10
    nsteps = 2 * patience
    ils = nk.callbacks.InvalidLossStopping(patience=patience)

    driver.run(nsteps, callback=ils)
    assert driver.step_count == nsteps

    params = flax.core.unfreeze(driver.state.parameters)
    params["visible_bias"] = np.inf * params["visible_bias"]
    if isinstance(driver, nkx.driver.TDVP):
        driver._integrator._state = driver._integrator._state.replace(y=params)
    driver.state.parameters = params
    driver.reset()

    driver.run(nsteps, callback=ils)
    assert ils._last_valid_iter == 0
    assert driver.step_count == patience


def test_invalid_loss_stopping_correct_interval():
    patience = 4
    cb = nk.callbacks.InvalidLossStopping(patience=patience)

    driver = nk.driver.AbstractVariationalDriver(
        None, None, minimized_quantity_name="loss"
    )

    log_data = {}
    cb(0, log_data, driver)
    assert cb._last_valid_iter == 0

    driver._loss_stats = nk.stats.Stats(mean=np.array(1.0))
    assert cb(2, log_data, driver)
    assert cb._last_valid_iter == 0

    driver._step_count = 2
    assert cb(None, log_data, driver)
    assert cb._last_valid_iter == 2

    driver._step_count = 3
    driver._loss_stats = nk.stats.Stats(mean=np.nan)
    assert cb(None, log_data, driver)
    assert cb._last_valid_iter == 2

    driver._step_count = 4
    assert cb(None, log_data, driver)
    assert cb._last_valid_iter == 2

    driver._step_count = 8
    assert not cb(None, log_data, driver)
    assert cb._last_valid_iter == 2


def test_convergence_stopping():
    loss_values = [10] + [9] * 12 + [1] * 4
    es = nk.callbacks.ConvergenceStopping(target=9.0, patience=10, smoothing_window=1)
    driver = DummyDriver()
    for step in range(len(loss_values)):
        print(es)
        if not es(step, {"loss": DummyLogEntry(loss_values[step])}, driver):
            break

    assert step == 11

    es = nk.callbacks.ConvergenceStopping(target=9.0, patience=10, smoothing_window=3)
    driver = DummyDriver()
    for step in range(len(loss_values)):
        print(es)
        if not es(step, {"loss": DummyLogEntry(loss_values[step])}, driver):
            break

    assert step == 13
