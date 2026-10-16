import math
from typing import Any

from tqdm.auto import tqdm

import jax

from netket.utils import struct

from netket._src.callbacks.base import AbstractCallback


def _elapsed_time(driver, t_start: float, T_final: float) -> float:
    """Time elapsed since `t_start`, at most `T_final`."""
    return min(float(driver.t) - t_start, T_final)


class TimeProgressBarCallback(AbstractCallback, mutable=True):
    """
    Progress bar displaying physical time for dynamics drivers.

    Tracks the time elapsed since the start of the run, ``driver.t - t_start ∈
    [0, T_final]``, instead of step count. ``t_start`` is ``driver._t_start`` if
    the driver sets it (as the TDVP drivers do), and 0 otherwise.
    Used automatically by :class:`~netket.driver.AbstractDynamicsDriver`
    when :meth:`run` is called with a float time argument.
    """

    _T_final: float = struct.field(pytree_node=False, serialize=False)
    _t_start: float = struct.field(pytree_node=False, serialize=False)
    _pbar: Any = struct.field(pytree_node=False, serialize=False)
    _leave: bool = struct.field(pytree_node=False, serialize=False)

    def __init__(self, T_final: float, leave: bool = True):
        self._T_final = T_final
        self._t_start = 0.0
        self._pbar = None
        self._leave = leave

    def on_run_start(self, step, driver):
        # Count from the time the run started at, even if another callback
        # changes the time (e.g. when resuming from a checkpoint).
        t_start = getattr(driver, "_t_start", None)
        self._t_start = 0.0 if t_start is None else float(t_start)

        dt = float(driver.dt)
        n_dec = max(4, math.ceil(-math.log10(dt))) if dt > 0 else 4
        bar_format = (
            f"{{l_bar}}{{bar}}| {{n:.{n_dec}f}}/{{total:.{n_dec}f}}"
            " [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        self._pbar = tqdm(
            total=self._T_final,
            unit="t",
            disable=not jax.process_index() == 0,
            dynamic_ncols=True,
            leave=self._leave,
            bar_format=bar_format,
        )
        self._pbar.update(_elapsed_time(driver, self._t_start, self._T_final))
        self._pbar.unpause()
        self._pbar.refresh()

    def before_parameter_update(self, step, log_data, driver):
        if driver._loss_stats is not None:
            self._pbar.set_postfix_str(
                driver._loss_name + "=" + str(driver._loss_stats)
            )

    def on_step_end(self, step, log_data, driver):
        self._pbar.update(
            _elapsed_time(driver, self._t_start, self._T_final) - self._pbar.n
        )
        self._pbar.refresh()

    def on_run_end(self, step, driver):
        if self._pbar is not None:
            self._pbar.update(
                _elapsed_time(driver, self._t_start, self._T_final) - self._pbar.n
            )
            self._pbar.close()
            self._pbar = None

    def on_run_error(self, step, error, driver):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


class ProgressBarCallback(AbstractCallback, mutable=True):
    _leave: bool = struct.field(pytree_node=False, serialize=False)

    _n_steps: int = struct.field(pytree_node=False, serialize=False)
    _start_step: int = struct.field(pytree_node=False, serialize=False)

    _pbar: Any = struct.field(pytree_node=False, serialize=False)

    def __init__(self, n_steps: int, leave: bool = True):
        self._pbar = None
        self._n_steps = n_steps
        self._leave = leave

    def on_run_start(self, step, driver):
        # Count from the step the run started at, even if another callback
        # changes the step count (e.g. when resuming from a checkpoint).
        self._start_step = getattr(driver, "_start_step", None)
        if self._start_step is None:
            self._start_step = driver.step_count

        self._pbar = tqdm(
            total=self._n_steps,
            disable=not jax.process_index() == 0,
            dynamic_ncols=True,
            leave=self._leave,
        )
        self._pbar.update(driver.step_count - self._start_step)
        self._pbar.unpause()
        self._pbar.refresh()

    def before_parameter_update(self, step, log_data, driver):
        # if the cost-function is defined then report it in the progress bar
        if driver._loss_stats is not None:
            self._pbar.set_postfix_str(
                driver._loss_name + "=" + str(driver._loss_stats)
            )

    def on_step_end(self, step, log_data, driver):
        step_value = driver.step_count - self._start_step
        self._pbar.update(step_value - self._pbar.n)
        self._pbar.refresh()

    def on_run_end(self, step, driver):
        step_value = driver.step_count - self._start_step
        if self._pbar is not None:
            self._pbar.update(step_value - self._pbar.n)
            self._pbar.refresh()
            self._pbar.close()
            self._pbar = None

    def on_run_error(self, step, error, driver):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
