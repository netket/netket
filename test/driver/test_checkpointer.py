import jax
import jax.numpy as jnp
import numpy as np
import pytest

import netket as nk
from netket._src.callbacks.legacy_wrappers import LegacyLoggerWrapper
from netket.callbacks import AbstractCallback, StopRun
from netket.utils import struct

from .. import common

pytestmark = common.skipif_distributed


def _system(mc=False):
    hi = nk.hilbert.Spin(0.5, 4)
    ha = nk.operator.IsingJax(hi, nk.graph.Chain(4), h=1.0)
    model = nk.models.RBM(alpha=1, param_dtype=float)
    if mc:
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=4)
        vs = nk.vqs.MCState(sampler, model, n_samples=64, seed=12, sampler_seed=13)
    else:
        vs = nk.vqs.FullSumState(hi, model, seed=12)
    return ha, vs


def _vmc(callbacks=(), *, mc=False, sr=False):
    ha, vs = _system(mc)
    if sr:
        return nk.driver.VMC_SR(
            ha,
            nk.optimizer.Sgd(0.05),
            diag_shift=0.1,
            momentum=0.9,
            variational_state=vs,
            callbacks=callbacks,
        )
    return nk.driver.VMC(
        ha, nk.optimizer.Sgd(0.05), variational_state=vs, callbacks=callbacks
    )


def _assert_same(got, want):
    for x, y in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        np.testing.assert_array_equal(x, y)


class _Recorder(AbstractCallback):
    """Appends the hooks it gets to ``events``; raises ``error`` from ``hook``."""

    events: list = struct.field(pytree_node=False, serialize=False)
    _hook: str | None = struct.field(pytree_node=False, serialize=False)
    _error: Exception | None = struct.field(pytree_node=False, serialize=False)

    def __init__(self, events, hook=None, error=None):
        self.events = events
        self._hook = hook
        self._error = error

    def _record(self, hook, *args):
        self.events.append((hook, *args))
        if hook == self._hook:
            raise self._error

    def on_run_prep_allocate(self, driver):
        self._record("on_run_prep_allocate")

    def on_checkpoint_restore(self, step, driver):
        self._record("on_checkpoint_restore", step)

    def on_run_start(self, step, driver):
        self._record("on_run_start", step)

    def on_step_start(self, step, log_data, driver):
        self._record("on_step_start", step)

    def on_step_end(self, step, log_data, driver):
        self._record("on_step_end", step)

    def on_checkpoint_save(self, step, driver):
        self._record("on_checkpoint_save", step)

    def on_run_end(self, step, driver):
        self._record("on_run_end", step)

    def on_run_error(self, step, error, driver):
        self._record("on_run_error", step)


class _StartCounter(AbstractCallback):
    """A persistent callback that changes its saved state in ``on_step_start``
    and stops the run after five started steps."""

    n: int = struct.field(pytree_node=False, serialize=True, default=0)

    def __init__(self):
        self.n = 0

    def on_step_start(self, step, log_data, driver):
        self.n += 1

    def on_step_end(self, step, log_data, driver):
        if self.n >= 5:
            raise StopRun("five started steps")


class _Boom(Exception):
    pass


class _CrashAt(AbstractCallback):
    _at: int = struct.field(pytree_node=False, serialize=False)

    def __init__(self, at):
        self._at = at

    def on_compute_update_start(self, step, log_data, driver):
        if step == self._at:
            raise _Boom("simulated interruption")


def _crash_then_resume(crashed, resumed, callbacks=list, **run):
    """Runs ``crashed`` (``run`` are the arguments), crashing in step 3 after the
    checkpoint of step 2, then ``resumed`` from that checkpoint; ``callbacks()``
    gives each run its own per-run callbacks."""
    ckpt = common.MemoryCheckpointer(every=2)
    with pytest.raises(_Boom):
        crashed.run(
            callback=[*callbacks(), _CrashAt(3)],
            checkpointer=ckpt,
            show_progress=False,
            **run,
        )
    resumed.run(callback=callbacks(), checkpointer=ckpt, show_progress=False, **run)
    return ckpt


def test_the_driver_runs_the_protocol_at_step_boundaries():
    events = []
    ckpt = common.MemoryCheckpointer(every=2, events=events)
    _vmc().run(3, callback=_Recorder(events), checkpointer=ckpt, show_progress=False)

    def save_point(step, saved, last=False):
        asked = [("should_stop", step), ("should_save", step, last)]
        if saved:
            asked += [("on_checkpoint_save", step), ("save", step)]
        return asked

    def step(s):
        return [("on_step_start", s), ("on_step_end", s)]

    # A fresh start: nothing restored, so no on_checkpoint_restore.
    assert events == [
        ("on_run_prep_allocate",),
        ("restore", 0, 3),
        ("on_run_start", 0),
        *step(0),
        *save_point(1, saved=False),
        *step(1),
        *save_point(2, saved=True),
        *step(2),
        *save_point(3, saved=True, last=True),
        ("wait",),
        ("on_run_end", 3),
        ("close",),
    ]
    assert ckpt.cause is None

    # Run again: it restores the last checkpoint, then calls on_checkpoint_restore
    # once, before on_run_start; the run is done.
    events.clear()
    _vmc().run(3, callback=_Recorder(events), checkpointer=ckpt, show_progress=False)
    assert events == [
        ("on_run_prep_allocate",),
        ("restore", 0, 3),
        ("on_checkpoint_restore", 3),
        ("on_run_start", 3),
        ("wait",),
        ("on_run_end", 3),
        ("close",),
    ]

    # Without a checkpointer, no checkpoint hook is called.
    events.clear()
    _vmc().run(3, callback=_Recorder(events), show_progress=False)
    assert events == [
        ("on_run_prep_allocate",),
        ("on_run_start", 0),
        *step(0),
        *step(1),
        *step(2),
        ("on_run_end", 3),
    ]


class _CheapInit(AbstractCallback):
    """Pattern 1: sets the parameters in on_run_prep_allocate; a restore then
    overwrites them, so the init only counts on a fresh start."""

    def on_run_prep_allocate(self, driver):
        driver.state.parameters = jax.tree.map(
            lambda p: jnp.full_like(p, 0.1), driver.state.parameters
        )


def test_a_cheap_init_counts_only_on_a_fresh_start():
    reference, resumed = _vmc(), _vmc()
    reference.run(6, callback=_CheapInit(), show_progress=False)
    _crash_then_resume(_vmc(), resumed, lambda: [_CheapInit()], n_iter=6)
    _assert_same(resumed.state.parameters, reference.state.parameters)


class _ExpensiveInit(AbstractCallback):
    """Pattern 2: thermalises in on_run_start, unless the run was restored (noted
    in on_checkpoint_restore, forgotten in on_run_prep_allocate at each run)."""

    inits: int = struct.field(pytree_node=False, serialize=False, default=0)
    _restored: bool = struct.field(pytree_node=False, serialize=False, default=False)

    def __init__(self):
        self.inits = 0
        self._restored = False

    def on_run_prep_allocate(self, driver):
        self._restored = False

    def on_checkpoint_restore(self, step, driver):
        self._restored = True

    def on_run_start(self, step, driver):
        if not self._restored:
            self.inits += 1
            driver.state.sample()


def test_an_expensive_init_runs_only_on_a_fresh_start():
    inits = []

    def expensive_init():
        inits.append(_ExpensiveInit())
        return [inits[-1]]

    reference, resumed = _vmc(mc=True), _vmc(mc=True)
    reference.run(6, callback=expensive_init(), show_progress=False)
    _crash_then_resume(_vmc(mc=True), resumed, expensive_init, n_iter=6)

    # Once uninterrupted, once crashed, not on the resume.
    assert [init.inits for init in inits] == [1, 1, 0]
    _assert_same(resumed.state.parameters, reference.state.parameters)
    _assert_same(resumed.state.sampler_state, reference.state.sampler_state)
    # A later run that restores nothing is a fresh start again.
    resumed.run(2, callback=inits[-1], show_progress=False)
    assert inits[-1].inits == 1


def test_a_checkpoint_is_read_without_a_driver_and_seeds_a_new_run():
    ckpt = common.MemoryCheckpointer(every=2)
    old = _vmc((_StartCounter(),), mc=True)
    old.run(4, checkpointer=ckpt, show_progress=False)

    checkpoint = ckpt.read()
    assert checkpoint.step == ckpt.latest_step() == 4
    assert set(checkpoint.state) == {"driver", "variational_state", "callbacks"}
    assert int(checkpoint.state["driver"]["_step_count"]) == 4
    assert checkpoint.state["callbacks"] == [{"n": 4}]

    # Another optimizer, no callbacks, twice the chains and samples.
    ha, vs = _system()
    sampler = nk.sampler.MetropolisLocal(vs.hilbert, n_chains=8)
    vs = nk.vqs.MCState(sampler, vs.model, n_samples=128, seed=3, sampler_seed=4)
    new = nk.driver.VMC(ha, nk.optimizer.Adam(0.01), variational_state=vs)
    nk.checkpoint.load_checkpoint_state(
        new, checkpoint.state, only=("variational_state/variables",)
    )
    _assert_same(new.state.parameters, old.state.parameters)
    assert new.step_count == 0 and new.state.n_samples == 128
    new.run(2, show_progress=False)


class _ParameterSum(AbstractCallback):
    """A persistent callback saving a buffer shaped like the parameters: the sum
    of the parameters at the end of every step."""

    total: dict | None = struct.field(pytree_node=False, serialize=True, default=None)

    def __init__(self):
        self.total = None

    def on_run_prep_allocate(self, driver):
        if self.total is None:
            self.total = jax.tree.map(jnp.zeros_like, driver.state.parameters)

    def on_step_end(self, step, log_data, driver):
        self.total = jax.tree.map(jnp.add, self.total, driver.state.parameters)


def test_a_resumed_run_replays_exactly():
    # Persistent callbacks: one changes its saved state in on_step_start (and stops
    # the run), one saves a buffer shaped like the parameters, allocated before the
    # restore (allocated lazily, it would be missing from what is restored into).
    def persistent():
        return (_StartCounter(), _ParameterSum())

    reference = _vmc(persistent(), mc=True)
    reference.run(10, show_progress=False)
    counter, total = persistent()
    resumed = _vmc((counter, total), mc=True)
    state = resumed.state
    _crash_then_resume(_vmc(persistent(), mc=True), resumed, n_iter=10)

    # Restored in place: the driver keeps its variational state and callbacks.
    assert resumed.state is state
    assert resumed.callbacks[0] is counter and resumed.callbacks[1] is total
    assert (resumed.step_count, counter.n) == (5, 5)
    _assert_same(total.total, reference.callbacks[1].total)
    _assert_same(resumed.state.parameters, reference.state.parameters)
    _assert_same(resumed.state.sampler_state, reference.state.sampler_state)


def test_auto_chunk_size_does_not_change_what_a_resumed_run_computes():
    reference, resumed = _vmc(mc=True, sr=True), _vmc(mc=True, sr=True)
    reference.run(6, show_progress=False)
    _crash_then_resume(
        _vmc(mc=True, sr=True),
        resumed,
        lambda: [nk.callbacks.AutoChunkSize()],
        n_iter=6,
    )
    _assert_same(resumed.state.parameters, reference.state.parameters)
    _assert_same(resumed.state.sampler_state, reference.state.sampler_state)
    _assert_same(resumed._old_updates, reference._old_updates)


def test_an_early_stop_saves_the_step_it_stopped_after():
    ckpt = common.MemoryCheckpointer(every=100)
    stopper = _StartCounter()
    driver = _vmc((stopper,))
    driver.run(10, checkpointer=ckpt, show_progress=False)

    # Stopped at the end of step 4: 5 updates, saved as step 5.
    assert driver.step_count == 5 and sorted(ckpt.checkpoints) == [5]
    assert str(ckpt.cause) == "five started steps"
    restored = _vmc((_StartCounter(),))
    nk.checkpoint.load_checkpoint_state(restored, ckpt.read(5).state)
    reference = _vmc()
    reference.run(5, show_progress=False)
    assert restored.step_count == 5
    _assert_same(restored.state.parameters, reference.state.parameters)


@pytest.mark.parametrize(
    "hook", ["on_run_start", "on_step_start", "on_checkpoint_save", "on_run_end"]
)
def test_a_stop_run_raised_elsewhere_is_an_error(hook):
    events = []
    ckpt = common.MemoryCheckpointer(every=1, events=events)
    with pytest.raises(RuntimeError, match="StopRun may only be returned"):
        _vmc().run(
            3,
            callback=_Recorder(events, hook, StopRun("x")),
            checkpointer=ckpt,
            show_progress=False,
        )
    assert "on_run_error" in [e[0] for e in events]
    assert isinstance(ckpt.cause, RuntimeError)


def test_the_checkpointer_stops_the_run_after_saving():
    events = []
    ckpt = common.MemoryCheckpointer(every=100, stop_at=3, events=events)
    driver = _vmc()
    driver.run(10, callback=_Recorder(events), checkpointer=ckpt, show_progress=False)

    assert driver.step_count == 3 and sorted(ckpt.checkpoints) == [3]
    assert events[-7:] == [
        ("should_stop", 3),
        ("should_save", 3, True),
        ("on_checkpoint_save", 3),
        ("save", 3),
        ("wait",),
        ("on_run_end", 3),
        ("close",),
    ]
    assert str(ckpt.cause) == "stop at step 3"


class _FailingCheckpointer(common.MemoryCheckpointer):
    """Raises ``error`` from its ``method`` (wait or close), after doing its job."""

    def __init__(self, method, error, **kwargs):
        super().__init__(**kwargs)
        self.method, self.error = method, error

    def wait(self):
        super().wait()
        if self.method == "wait":
            raise self.error

    def close(self, cause=None):
        super().close(cause)
        if self.method == "close":
            raise self.error


@pytest.mark.parametrize("error", [OSError("disk full"), KeyboardInterrupt()])
@pytest.mark.parametrize(
    ("where", "step"), [("on_checkpoint_save", 2), ("wait", 3), ("on_run_end", 3)]
)
def test_an_error_while_saving_or_ending_goes_to_on_run_error(where, step, error):
    # A failed flush saves nothing; a failed background save surfaces at wait; a
    # Ctrl-C there or in a finaliser is an interrupted run.
    events = []
    ckpt = _FailingCheckpointer(where, error, every=2, events=events)
    with pytest.raises(type(error)):
        _vmc().run(
            3,
            callback=_Recorder(events, where, error),
            checkpointer=ckpt,
            show_progress=False,
        )
    assert events[-2:] == [("on_run_error", step), ("close",)]
    assert ckpt.cause is error
    if where == "on_checkpoint_save":
        assert ckpt.checkpoints == {}


def test_a_failed_close_is_raised_with_the_error_of_the_run_as_its_context():
    def failing():
        return _FailingCheckpointer("close", RuntimeError("close failed"))

    error_run = _Recorder([], "on_step_end", OSError("disk full"))
    with pytest.raises(RuntimeError, match="close failed") as error:
        _vmc().run(3, callback=error_run, checkpointer=failing(), show_progress=False)
    assert isinstance(error.value.__context__, OSError)

    # Raised, also in a caller's except block (it was once noted there, and lost).
    try:
        raise LookupError("the caller's")
    except LookupError as caller_error:
        with pytest.raises(RuntimeError, match="close failed"):
            _vmc().run(3, checkpointer=failing(), show_progress=False)
        assert not getattr(caller_error, "__notes__", None)


@pytest.mark.parametrize(
    "callback",
    [
        lambda step, log, driver: True,
        nk.logging.RuntimeLog(),
        LegacyLoggerWrapper(nk.logging.RuntimeLog()),
        common.MemoryCheckpointer(),
    ],
)
def test_persistent_callbacks_are_checked_at_construction(callback):
    with pytest.raises(TypeError, match="persistent callback|checkpointer="):
        _vmc((callback,))


def test_persistent_callbacks_are_fixed():
    driver = _vmc((_StartCounter(),))
    with pytest.raises(AttributeError):
        driver.callbacks = ()
    with pytest.raises(TypeError, match="AbstractCheckpointer"):
        driver.run(2, checkpointer=nk.callbacks.EarlyStopping(), show_progress=False)


def test_a_string_out_appends_when_resuming(tmp_path):
    prefix = str(tmp_path / "log")
    _crash_then_resume(_vmc(), _vmc(), n_iter=6, out=prefix, write_every=1)
    energy = nk.logging.RuntimeLog.deserialize(prefix + ".log")["Energy"]
    assert list(energy.iters) == list(range(6))


class _Euler(nk.driver.AbstractDynamicsDriver):
    """Imaginary-time Euler steps of a full-sum state."""

    _ham: nk.operator.AbstractOperator = struct.field(
        pytree_node=False, serialize=False
    )

    def __init__(self, ham, vs, callbacks=()):
        super().__init__(vs, callbacks=callbacks)
        self._ham = ham

    @property
    def dt(self):
        return 0.05

    def compute_loss_and_update(self):
        energy, grad = self.state.expect_and_grad(self._ham)
        return energy, jax.tree.map(lambda g: -self.dt * g, grad)


def test_a_dynamics_driver_resumes_from_its_checkpointer():
    reference, resumed = _Euler(*_system()), _Euler(*_system())
    reference.run(n_iter=6, show_progress=False)
    _crash_then_resume(_Euler(*_system()), resumed, n_iter=6)
    assert resumed.step_count == 6 and resumed.t == reference.t
    _assert_same(resumed.state.parameters, reference.state.parameters)


def _tdvp():
    ha, vs = _system()
    return nk.experimental.TDVP(
        ha, vs, nk.experimental.dynamics.Euler(dt=0.01), propagation_type="imag"
    )


def test_a_tdvp_run_resumes_from_its_checkpointer(tmp_path):
    reference, resumed = _tdvp(), _tdvp()
    log = nk.logging.RuntimeLog()
    reference.run(0.06, out=log, show_progress=False)
    prefix = str(tmp_path / "log")
    _crash_then_resume(_tdvp(), resumed, T=0.06, out=prefix)

    assert resumed.step_count == reference.step_count
    np.testing.assert_array_equal(resumed.t, reference.t)
    _assert_same(resumed.state.parameters, reference.state.parameters)
    # The log has every stopping time once, the replayed ones included.
    times = nk.logging.RuntimeLog.deserialize(prefix + ".log")["t"].iters
    np.testing.assert_array_equal(times, log.data["t"].iters)
