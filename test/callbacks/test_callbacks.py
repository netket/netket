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

    return nk.driver.VMC(hamiltonian=ha, variational_state=vs, optimizer=op)


def _tdvp(n_iter=20):
    hi = nk.hilbert.Spin(s=0.5) ** L

    ma = nk.models.RBM(alpha=1)
    # rescale so that dt=1.0
    ha = 1e-2 * nk.operator.IsingJax(hi, nk.graph.Hypercube(length=L, n_dim=1), h=1.0)
    sa = nk.sampler.MetropolisLocal(hi)
    vs = nk.vqs.MCState(sa, ma, n_samples=512, seed=SEED)

    ode_solver = nkx.dynamics.RK4(dt=1.0)
    solv = nk.optimizer.solver.svd(rcond=1e-5)

    return nkx.TDVP(
        operator=ha, variational_state=vs, ode_solver=ode_solver, linear_solver=solv
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
        try:
            es.on_step_end(step, {"loss": DummyLogEntry(loss_values[step])}, driver)
        except nk.callbacks.StopRun:
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
        try:
            es.on_step_end(step, {"loss": DummyLogEntry(loss_values[step])}, driver)
        except nk.callbacks.StopRun:
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
        try:
            es.on_step_end(step, {"loss": DummyLogEntry(loss_values[step])}, driver)
        except nk.callbacks.StopRun:
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
        try:
            es.on_step_end(step, {"loss": DummyLogEntry(loss_values[step])}, driver)
        except nk.callbacks.StopRun:
            break

    assert step == 12
    assert es._best_iter == 1
    assert es._best_val == 10.0

    es = nk.callbacks.EarlyStopping(patience=10, min_reldelta=1e-4)
    driver = DummyDriver()
    for step in range(len(loss_values)):
        try:
            es.on_step_end(step, {"loss": DummyLogEntry(loss_values[step])}, driver)
        except nk.callbacks.StopRun:
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
        try:
            es.on_step_end(step, {"loss": DummyLogEntry(loss_values[step])}, driver)
        except nk.callbacks.StopRun:
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
        try:
            es.on_step_end(step, {"loss": DummyLogEntry(loss_values[step])}, driver)
        except nk.callbacks.StopRun:
            break
    assert step == 22
    assert es._best_iter == 18
    assert es._best_val == 9.0


@pytest.mark.parametrize("driver", [_vmc()])
def test_invalid_loss_stopping(driver):
    patience = 10
    nsteps = 2 * patience
    ils = nk.callbacks.InvalidLossStopping(patience=patience)

    driver.run(nsteps, callback=ils)
    assert driver.step_count == nsteps
    step_count_before_invalid = driver.step_count

    params = flax.core.unfreeze(driver.state.parameters)
    params["visible_bias"] = np.inf * params["visible_bias"]
    if isinstance(driver, nkx.driver.TDVP):
        driver._integrator._state = driver._integrator._state.replace(y=params)
    driver.state.parameters = params
    driver.reset_step()

    driver.run(nsteps, callback=ils)
    # The driver should stop early after approximately patience invalid steps.
    # The exact count differs by ±1 depending on whether step_count is updated
    # before or after the callback fires (TDVP vs VMC).
    assert (
        step_count_before_invalid + patience - 1
        <= driver.step_count
        <= step_count_before_invalid + patience
    )


class FakeState:
    parameters = {}


def test_invalid_loss_stopping_correct_interval():
    patience = 4
    cb = nk.callbacks.InvalidLossStopping(patience=patience)

    driver = nk.driver.AbstractVariationalDriver(
        FakeState(), nk.optimizer.Sgd(0.01), minimized_quantity_name="loss"
    )

    log_data = {}
    cb.on_step_end(0, log_data, driver)
    assert cb._last_valid_iter == 0

    driver._loss_stats = nk.stats.Stats(mean=np.array(1.0))
    cb.on_step_end(2, log_data, driver)
    assert cb._last_valid_iter == 0

    driver._step_count = 2
    cb.on_step_end(None, log_data, driver)
    assert cb._last_valid_iter == 2

    driver._step_count = 3
    driver._loss_stats = nk.stats.Stats(mean=np.nan)
    cb.on_step_end(None, log_data, driver)
    assert cb._last_valid_iter == 2

    driver._step_count = 4
    cb.on_step_end(None, log_data, driver)
    assert cb._last_valid_iter == 2

    driver._step_count = 8
    with pytest.raises(nk.callbacks.StopRun):
        cb.on_step_end(None, log_data, driver)
    assert cb._last_valid_iter == 2


def test_invalid_loss_stopping_vector_valued():
    # A vector-valued loss (e.g. a foundation ReplicaStats has one mean per
    # anchor) must not crash the finiteness check: `not np.isfinite(loss)` is
    # ambiguous for a non-scalar array. The run stops iff any component is
    # non-finite, and only after `patience` consecutive invalid steps.
    patience = 4
    cb = nk.callbacks.InvalidLossStopping(patience=patience)

    driver = nk.driver.AbstractVariationalDriver(
        FakeState(), nk.optimizer.Sgd(0.01), minimized_quantity_name="loss"
    )
    log_data = {}

    # All-finite vector: valid, no crash.
    driver._step_count = 0
    driver._loss_stats = nk.stats.Stats(mean=np.array([1.0, 2.0, 3.0]))
    cb.on_step_end(None, log_data, driver)
    assert cb._last_valid_iter == 0

    # One non-finite component: invalid, but within patience so no stop yet.
    driver._step_count = 1
    driver._loss_stats = nk.stats.Stats(mean=np.array([1.0, np.nan, 3.0]))
    cb.on_step_end(None, log_data, driver)
    assert cb._last_valid_iter == 0

    # Still invalid after `patience` steps -> stop.
    driver._step_count = patience + 1
    with pytest.raises(nk.callbacks.StopRun):
        cb.on_step_end(None, log_data, driver)


def test_save_variational_state_max_to_keep(tmp_path):
    pytest.importorskip("nqxpack")

    interval = 5
    n_iter = 20
    max_to_keep = 2
    root = "state"

    cb = nk.logging.SaveVariationalState(
        path=tmp_path, interval=interval, max_to_keep=max_to_keep
    )

    # A foreign file that does not match the {root}_{step}.nk pattern must never
    # be deleted by the pruning logic.
    guard = tmp_path / f"{root}_keepme.nk"
    guard.write_text("foreign")

    vmc = _vmc()
    vmc.run(n_iter, callback=cb)

    saved = sorted(p.name for p in tmp_path.glob(f"{root}_*.nk"))

    # Only the most recent `max_to_keep` checkpoints survive, plus the guard file.
    assert guard.exists()
    assert f"{root}_keepme.nk" in saved
    checkpoints = [name for name in saved if name != f"{root}_keepme.nk"]
    assert len(checkpoints) == max_to_keep
    # The newest saves are the last interval step (15) and the final step (20).
    assert checkpoints == [f"{root}_00015.nk", f"{root}_00020.nk"]


def test_convergence_stopping():
    loss_values = [10] + [9] * 12 + [1] * 4
    es = nk.callbacks.ConvergenceStopping(target=9.0, patience=10, smoothing_window=1)
    driver = DummyDriver()
    for step in range(len(loss_values)):
        print(es)
        try:
            es.on_step_end(step, {"loss": DummyLogEntry(loss_values[step])}, driver)
        except nk.callbacks.StopRun:
            break

    assert step == 11

    es = nk.callbacks.ConvergenceStopping(target=9.0, patience=10, smoothing_window=3)
    driver = DummyDriver()
    for step in range(len(loss_values)):
        print(es)
        try:
            es.on_step_end(step, {"loss": DummyLogEntry(loss_values[step])}, driver)
        except nk.callbacks.StopRun:
            break

    assert step == 13


@pytest.mark.parametrize("restore_before_bar", [True, False])
def test_progress_bar_after_resume(restore_before_bar):
    # run(8) called at step 0, with a checkpoint restoring step 5 either before or
    # after the progress bar starts.
    driver = DummyDriver()
    driver.step_count = 0
    driver._start_step = 0
    driver._loss_stats = None

    pb = nk._src.callbacks.progressbar.ProgressBarCallback(8)
    if restore_before_bar:
        driver.step_count = 5
    pb.on_run_start(driver.step_count, driver)
    driver.step_count = 5
    assert pb._pbar.n == (5 if restore_before_bar else 0)

    for step in range(5, 8):
        driver.step_count = step + 1
        pb.on_step_end(step, {}, driver)
    assert pb._pbar.n == 8
    pb.on_run_end(driver.step_count, driver)


def test_progress_bar_with_its_own_size():
    # A bar bigger than the run, created by the user: run(10) shows 10/20.
    driver = DummyDriver()
    driver.step_count = 0
    driver._start_step = 0
    driver._loss_stats = None

    pb = nk._src.callbacks.progressbar.ProgressBarCallback(20)
    pb.on_run_start(0, driver)
    assert pb._pbar.n == 0
    for step in range(10):
        driver.step_count = step + 1
        pb.on_step_end(step, {}, driver)
    assert pb._pbar.n == 10
    pb.on_run_end(driver.step_count, driver)
