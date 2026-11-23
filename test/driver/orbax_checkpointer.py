import signal
import threading
from pathlib import Path

import jax
import numpy as np
import pytest
from jax.experimental import multihost_utils

from netket._src import distributed
from netket._src.callbacks.base import StopRun
from netket._src.driver.checkpointer import (
    AbstractCheckpointer,
    Checkpoint,
    checkpoint_state,
    flatten_state,
    group_state,
)

# Skips the tests that use it if orbax is missing.
ocp = pytest.importorskip("orbax.checkpoint")


class Checkpointer(AbstractCheckpointer):
    """
    A checkpointer for the tests, saving a run to ``path`` with orbax: every
    ``save_every`` steps and at its end, keeping the latest ``keep_last_k``
    checkpoints. On ``signal_number``, a first Ctrl-C or a preemption notice of
    jax, the run saves and stops after the current step, on every process. The
    checkpoints in ``path`` belong to one call of ``run``: resuming them from
    another one raises.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        save_every: int = 50,
        keep_last_k: int = 2,
        signal_number: int | None = signal.SIGTERM,
    ):
        if save_every < 1 or keep_last_k < 1:
            raise ValueError("save_every and keep_last_k must be at least 1.")
        # orbax needs an absolute path.
        self.path = Path(path).absolute()
        self._save_every = int(save_every)
        self._keep_last_k = int(keep_last_k)
        self._signal_number = signal_number
        self._manager = None
        self._run = {}
        self._restored_step = None
        self._last_saved_step = None
        # Not reset by a new run: the job was told to stop.
        self._signal_received = False
        self._preempted = False
        self._handlers = {}

    def _open(self):
        if self._manager is None:
            options = ocp.CheckpointManagerOptions(
                max_to_keep=self._keep_last_k, cleanup_tmp_directories=True
            )
            self._manager = ocp.CheckpointManager(
                self.path,
                options=options,
                item_handlers=ocp.PyTreeCheckpointHandler(),
            )
        return self._manager

    def restore(self, driver, *, start_step, target_step):
        self._run = {"start_step": int(start_step), "target_step": int(target_step)}
        self._restored_step = self._last_saved_step = None
        self._install_signal_handlers()

        checkpoint = self.read(target=checkpoint_state(driver))
        if checkpoint is None:
            return None
        run = {key: checkpoint.metadata.get(key) for key in self._run}
        if run != self._run or not start_step <= checkpoint.step <= target_step:
            raise ValueError(
                f"The checkpoint of step {checkpoint.step} in {self.path} belongs to "
                f"a run from step {run['start_step']} to {run['target_step']}, not "
                f"to this one, from step {start_step} to {target_step}: resume that "
                "run, or use another directory for this one."
            )
        self._restored_step = checkpoint.step
        state = checkpoint.state
        # The chunk size is a memory setting of this run, not part of its state.
        if "chunk_size" in state["variational_state"]:
            state["variational_state"]["chunk_size"] = driver.state.chunk_size
        return state

    def latest_step(self):
        # The processes may list the directory differently on a networked
        # filesystem: they agree on the latest step any of them sees.
        step = self._open().latest_step()
        steps = multihost_utils.process_allgather(
            np.asarray([-1 if step is None else step], dtype=np.int32)
        )
        return None if np.max(steps) < 0 else int(np.max(steps))

    def read(self, step=None, *, target=None):
        if step is None:
            step = self.latest_step()
            if step is None:
                return None
        manager = self._open()
        manager.wait_until_finished()
        args = None
        if target is not None:
            # Only the saved persistent callbacks that the driver has: one it has
            # but that was not saved fails to load, with a clear error.
            item = flatten_state(target)
            saved = manager.metadata(step).item_metadata["_callbacks"]
            callbacks = {k: v for k, v in item.pop("_callbacks").items() if k in saved}
            if callbacks:
                item["_callbacks"] = callbacks
            args = ocp.args.PyTreeRestore(item=item, partial_restore=True)
        state = manager.restore(step, args=args)
        state.setdefault("_callbacks", {})
        metadata = dict(manager.metadata(step).custom_metadata or {})
        return Checkpoint(step, group_state(state), metadata)

    def should_stop(self, step):
        if self._signal_number is not None and not self._preempted:
            # A collective: every process asks at every step.
            self._preempted = ocp.multihost.reached_preemption(step)
        if distributed.any_rank(self._signal_received or self._preempted):
            return StopRun(
                f"Stopped by a signal or a preemption notice at step {step}."
            )
        return None

    def should_save(self, step, *, last):
        due = last or step % self._save_every == 0
        return due and step not in (self._restored_step, self._last_saved_step)

    def save(self, step, state):
        # Written in the background: copy the numpy arrays, which the run may
        # change in place (jax arrays cannot change).
        state = jax.tree.map(
            lambda x: x.copy() if isinstance(x, np.ndarray) else x,
            flatten_state(state),
        )
        self._open().save(
            step,
            args=ocp.args.PyTreeSave(state),
            custom_metadata=self._run,
            force=True,
        )
        self._last_saved_step = step

    def wait(self):
        if self._manager is not None:
            self._manager.wait_until_finished()

    def close(self, cause=None):
        manager, self._manager = self._manager, None
        try:
            if manager is not None:
                manager.close()
        finally:
            self._restore_signal_handlers()

    def _install_signal_handlers(self):
        if self._signal_number is None:
            return
        if threading.current_thread() is not threading.main_thread():
            return

        def stop(signum, frame):
            # Only set a flag: the run stops at the end of the step.
            self._signal_received = True

        def first_interrupt(signum, frame):
            self._signal_received = True
            signal.signal(signal.SIGINT, self._handlers[signal.SIGINT])

        handlers = {signal.SIGINT: first_interrupt}
        # In a multi-process run, jax handles SIGTERM itself (installing a handler
        # here would replace its own): `should_stop` asks it for a preemption.
        if self._signal_number != signal.SIGTERM or not _jax_handles_sigterm():
            handlers[self._signal_number] = stop
        for signum, handler in handlers.items():
            self._handlers[signum] = signal.signal(signum, handler)

    def _restore_signal_handlers(self):
        handlers, self._handlers = self._handlers, {}
        for signum, handler in handlers.items():
            # None: a handler installed outside Python, which cannot be restored.
            if handler is not None:
                signal.signal(signum, handler)


def _jax_handles_sigterm() -> bool:
    return jax.distributed.is_initialized() and jax.config.jax_enable_preemption_service
