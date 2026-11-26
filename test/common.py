# File containing common commands for NetKet Test infrastructure

from typing import Any
from functools import partial

import pytest

import jax
import netket as nk
import numpy as np


_multiple_devices = jax.device_count() > 1

requires_multiple_devices = pytest.mark.skipif(
    not _multiple_devices, reason="Needs more than one device"
)
"""Use as a decorator to mark a test that only makes sense with several devices."""

requires_multiple_devices_single_process = pytest.mark.skipif(
    not (jax.local_device_count() > 1 and jax.process_count() == 1),
    reason="Needs several devices in a single process",
)
"""Use as a decorator to mark a test that requires multi-device sharding but is
incompatible with multi-process distributed execution (e.g. because it uses
process-local warnings or single-device reference meshes).
"""


def skipif_multiple_devices(reason: str):
    """Skip the decorated test when running on several devices, for code that is
    not supported there. A reason is mandatory."""
    return pytest.mark.skipif(_multiple_devices, reason=reason)


def xfail_multiple_devices(reason: str, strict: bool = True):
    """Mark the decorated test as known to fail on several devices. A reason is
    mandatory."""
    return pytest.mark.xfail(_multiple_devices, reason=reason, strict=strict)


class set_config:
    """
    Temporarily changes the value of the configuration `name`.

    Example:

    >>> with set_config("netket_experimental_disable_ode_jit", True):
    >>>     run_code

    """

    def __init__(self, name: str, value: Any):
        self._name = name.upper()
        self._value = value

    def __enter__(self):
        self._orig_value = nk.config.FLAGS[self._name]
        nk.config.update(self._name, self._value)

    def __exit__(self, exc_type, exc_value, traceback):
        nk.config.update(self._name, self._orig_value)


netket_experimental_fft_autocorrelation = partial(
    set_config, "NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION"
)


def hash_for_seed(obj):
    """
    Hash any object into an int that can be used in `np.random.seed`, and does not change between Python sessions.

    Args:
      obj: any object with `repr` defined to show its states.
    """

    bs = repr(obj).encode()
    out = 0
    for b in bs:
        out = (out * 256 + b) % 4294967291  # Output in [0, 2**32 - 1]
    return out


def named_parametrize(argname: str, values: list):
    param_values = [pytest.param(obj, id=f"{argname}={obj}") for obj in values]
    return pytest.mark.parametrize(argname, param_values)


def assert_allclose(x, y, **kwargs):
    from jax.experimental import multihost_utils

    if isinstance(x, jax.Array):
        if not x.is_fully_addressable:
            x = multihost_utils.process_allgather(x)
    if isinstance(y, jax.Array):
        if not y.is_fully_addressable:
            y = multihost_utils.process_allgather(y)
    np.testing.assert_allclose(x, y, **kwargs)


class MemoryCheckpointer(nk.experimental.checkpoint.AbstractCheckpointer):
    """
    Keeps the checkpoints of a run in ``checkpoints`` (step -> state), saving every
    ``every`` steps and where the run ends, and stopping the run at ``stop_at``.
    Appends what it is asked to ``events``.
    """

    def __init__(self, every=2, *, checkpoints=None, stop_at=None, events=None):
        self.every = every
        self.checkpoints = {} if checkpoints is None else checkpoints
        self.stop_at = stop_at
        self.events = [] if events is None else events

    def restore(self, driver, *, start_step, target_step):
        self.events.append(("restore", start_step, target_step))
        checkpoint = self.read(target=nk.checkpoint.checkpoint_state(driver))
        return None if checkpoint is None else checkpoint.state

    def latest_step(self):
        return max(self.checkpoints, default=None)

    def read(self, step=None, *, target=None):
        step = self.latest_step() if step is None else step
        if step is None:
            return None
        state = self.checkpoints[step]
        # Like orbax, read into a target only if it has the same structure.
        if target is not None and jax.tree.structure(state) != jax.tree.structure(
            target
        ):
            raise ValueError("The checkpoint does not match the driver's structure.")
        return nk.experimental.checkpoint.Checkpoint(step, state, {})

    def should_save(self, step, *, last):
        self.events.append(("should_save", step, last))
        return last or step % self.every == 0

    def save(self, step, state):
        self.events.append(("save", step))
        self.checkpoints[step] = state

    def should_stop(self, step):
        self.events.append(("should_stop", step))
        if step == self.stop_at:
            return nk.callbacks.StopRun(f"stop at step {step}")

    def wait(self):
        self.events.append(("wait",))

    def close(self, cause=None):
        self.events.append(("close",))
        self.cause = cause
