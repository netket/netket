import jax
import numpy as np
import pytest

import netket as nk
from netket.callbacks import AbstractCallback, StopRun
from netket.utils import struct

from .. import common


def _system():
    hi = nk.hilbert.Spin(0.5, 4)
    ha = nk.operator.IsingJax(hi, nk.graph.Chain(4), h=1.0)
    vs = nk.vqs.FullSumState(hi, nk.models.RBM(alpha=1, param_dtype=float), seed=12)
    return ha, vs


def _vmc():
    ha, vs = _system()
    return nk.driver.VMC(ha, nk.optimizer.Sgd(0.05), variational_state=vs)


class _Counter(AbstractCallback):
    """Counts the hooks it sees; stops the run at the end of step ``stop_at``:
    returns a StopRun, or raises it (the older way)."""

    calls: dict = struct.field(pytree_node=False, serialize=False)
    _stop_at: int | None = struct.field(pytree_node=False, serialize=False)
    _raises: bool = struct.field(pytree_node=False, serialize=False)

    def __init__(self, stop_at=None, raises=False):
        self.calls = {"on_step_end": 0, "on_run_end": 0, "on_run_error": 0}
        self._stop_at = stop_at
        self._raises = raises

    def on_step_end(self, step, log_data, driver):
        self.calls["on_step_end"] += 1
        if step == self._stop_at:
            stop = StopRun(f"{'raised' if self._raises else 'returned'} at step {step}")
            if self._raises:
                raise stop
            return stop

    def on_run_end(self, step, driver):
        self.calls["on_run_end"] += 1

    def on_run_error(self, step, error, driver):
        self.calls["on_run_error"] += 1


class _FailsAtRunEnd(_Counter):
    _error: Exception = struct.field(pytree_node=False, serialize=False)

    def __init__(self, error):
        super().__init__()
        self._error = error

    def on_run_end(self, step, driver):
        super().on_run_end(step, driver)
        raise self._error


@pytest.mark.parametrize(("stopper", "raises"), [(0, False), (1, True), (2, False)])
def test_stop_run_from_on_step_end_counts_the_step(stopper, raises):
    callbacks = [_Counter(2, raises) if i == stopper else _Counter() for i in range(3)]
    driver = _vmc()
    driver.run(10, callback=callbacks, show_progress=False)

    # Step 2 is complete: every callback saw its end, and it is counted.
    assert driver.step_count == 3
    assert [c.calls["on_step_end"] for c in callbacks] == [3, 3, 3]
    reference = _vmc()
    reference.run(3, show_progress=False)
    for got, want in zip(
        jax.tree.leaves(driver.state.parameters),
        jax.tree.leaves(reference.state.parameters),
        strict=True,
    ):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("first_raises", [False, True])
def test_the_first_stop_wins_returned_or_raised(first_raises, capsys):
    callbacks = [
        _Counter(),
        _Counter(2, raises=first_raises),
        _Counter(2, raises=not first_raises),
    ]
    driver = _vmc()
    driver.run(10, callback=callbacks, show_progress=False)

    assert driver.step_count == 3
    assert [c.calls["on_step_end"] for c in callbacks] == [3, 3, 3]
    first = "raised" if first_raises else "returned"
    out = capsys.readouterr().out
    assert f"Stopping early because of: {first} at step 2" in out


def test_a_legacy_callback_stops_the_run_after_the_update_of_its_step():
    # Called before the update, as before; its False stops the run after it.
    seen = []

    def legacy(step, log_data, driver):
        seen.append(step)
        return step < 1

    driver = _vmc()
    driver.run(10, callback=legacy, show_progress=False)
    assert seen == [0, 1] and driver.step_count == 2


def test_a_stop_on_one_process_stops_every_process():
    # Run with several processes (djaxrun): only process 0 asks to stop.
    callback = _Counter(2) if jax.process_index() == 0 else _Counter()
    driver = _vmc()
    driver.run(10, callback=callback, show_progress=False)
    assert driver.step_count == 3


@common.skipif_distributed
def test_a_legacy_callback_stops_a_tdvp_run_after_the_step_of_its_stopping_time():
    # Called at the stopping times, with the time; its False stops the run after
    # the step it was called in.
    seen = []

    def legacy(t, log_data, driver):
        seen.append(float(t))
        return len(seen) < 2

    driver = nk.experimental.TDVP(
        *_system(), nk.experimental.dynamics.Euler(dt=0.01), propagation_type="imag"
    )
    driver.run(0.06, callback=legacy, show_progress=False)
    assert driver.step_count == 2
    np.testing.assert_allclose(seen, [0.0, 0.01])
    np.testing.assert_allclose(driver.t, 0.02)


@pytest.mark.parametrize("stopped_run", [False, True])
@pytest.mark.parametrize(
    ("error", "raised"), [(StopRun("x"), RuntimeError), (OSError("disk full"), OSError)]
)
def test_an_error_in_on_run_end_goes_to_on_run_error(error, raised, stopped_run):
    # A StopRun raised there is a bug (the run has ended), an error like any other.
    callbacks = [_Counter(), _FailsAtRunEnd(error), _Counter()]
    if stopped_run:
        callbacks.append(_Counter(1))
    driver = _vmc()
    with pytest.raises(raised):
        driver.run(4, callback=callbacks, show_progress=False)

    assert driver.step_count == (2 if stopped_run else 4)
    assert [c.calls["on_run_end"] for c in callbacks] == [1, 1, 0, 0][: len(callbacks)]
    assert [c.calls["on_run_error"] for c in callbacks] == [1] * len(callbacks)
