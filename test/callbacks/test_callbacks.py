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


def test_earlystopping_with_patience():
    patience = 10
    es = nk.callbacks.EarlyStopping(patience=patience)
    es._best_val = -1e6
    vmc = _vmc()

    vmc.run(20, callback=es)

    assert vmc.step_count == patience


def test_earlystopping_baseline_with_patience():
    loss_values = np.array([11] + [10] * 12 + [9] * 4, dtype=float)
    loss_values[1:13] = 10.0 - 1e-3 * np.arange(12)

    # Because we have min_delta = min_rdelta = 0 this should not stop
    # however we do not drop under baseline in `patience` number of steps
    es = nk.callbacks.EarlyStopping(patience=10, baseline=9)
    driver = DummyDriver()
    for step in range(len(loss_values)):
        print(es)
        if not es(step, {"loss": DummyLogEntry(loss_values[step])}, driver):
            break

    assert step == 10
    assert es._best_iter == 10
    assert es._best_val == loss_values[10]


def test_earlystopping_with_delayed_start():
    loss_values = np.array([11] * 20 + [10] * 12 + [9] * 6, dtype=float)
    es = nk.callbacks.EarlyStopping(patience=10, start_from_step=9)
    # Until step 9, es._best_val is inf. In step 10 we have _best_patience_counter 1
    # In step 19 it is 10, the test self._best_patience_counter > self.patience
    # and not self._best_patience_counter >= self.patience
    # In step 20 we would fail, however loss_value drops to 10
    driver = DummyDriver()
    for step in range(len(loss_values)):
        print(es)
        if not es(step, {"loss": DummyLogEntry(loss_values[step])}, driver):
            break
    assert step == 31
    assert es._best_iter == 20
    assert es._best_val == 10.0


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


def test_earlystopping_baseline_with_patience_abstol_delayed_start():
    patience = 3
    start_from_step = 7
    loss_values = np.array([11] * 6 + [10] * 12 + [9] * 6, dtype=float)
    loss_values[7:19] = 10.0 - 1e-2 * np.arange(12)  # Note 1e-2
    loss_values[-6:] = 9 - 1e-3 * np.arange(6)  # Note 1e-3

    # We do not drop below baseline considering min_delta
    es = nk.callbacks.EarlyStopping(
        patience=patience, baseline=10, min_delta=1e-1, start_from_step=start_from_step
    )
    driver = DummyDriver()
    for step in range(len(loss_values)):
        print(es)
        if not es(step, {"loss": DummyLogEntry(loss_values[step])}, driver):
            break
    assert step == patience + start_from_step
    assert es._best_iter == 7
    assert es._best_val == 10.0

    # We drop below baseline and early stop because lack of convergence
    # Note smaller min_delta (1e-2 instead of 1e-1)
    es = nk.callbacks.EarlyStopping(
        patience=patience, baseline=10, min_delta=1e-2, start_from_step=start_from_step
    )
    driver = DummyDriver()
    for step in range(len(loss_values)):
        print(es)
        if not es(step, {"loss": DummyLogEntry(loss_values[step])}, driver):
            break
    assert step == 22
    assert es._best_iter == 18
    assert es._best_val == 9.0


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
