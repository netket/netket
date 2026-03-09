from typing import Any

from tqdm.auto import tqdm

import jax

from netket.utils import struct

from netket._src.callbacks.base import AbstractCallback


class TimeProgressBarCallback(AbstractCallback, mutable=True):
    """
    Progress bar displaying physical time for dynamics drivers.

    Tracks ``driver.t ∈ [0, T_final]`` instead of step count.
    Used automatically by :class:`~netket.driver.AbstractDynamicsDriver`
    when :meth:`run` is called with a float time argument.
    """

    _T_final: float = struct.field(pytree_node=False, serialize=False)
    _last_t: float = struct.field(pytree_node=False, serialize=False)
    _pbar: Any = struct.field(pytree_node=False, serialize=False)
    _leave: bool = struct.field(pytree_node=False, serialize=False)

    def __init__(self, T_final: float, leave: bool = True):
        self._T_final = T_final
        self._last_t = 0.0
        self._pbar = None
        self._leave = leave

    def on_run_start(self, step, driver):
        self._last_t = float(driver.t)
        self._pbar = tqdm(
            total=self._T_final,
            unit="t",
            disable=not jax.process_index() == 0,
            dynamic_ncols=True,
            leave=self._leave,
        )
        self._pbar.update(float(driver.t))
        self._pbar.unpause()
        self._pbar.refresh()

    def before_parameter_update(self, step, log_data, driver):
        if driver._loss_stats is not None:
            self._pbar.set_postfix_str(
                driver._loss_name + "=" + str(driver._loss_stats)
            )

    def on_step_end(self, step, log_data, driver):
        t = float(driver.t)
        self._pbar.update(t - self._last_t)
        self._last_t = t
        self._pbar.refresh()

    def on_run_end(self, step, driver):
        if self._pbar is not None:
            t = float(driver.t)
            self._pbar.update(t - self._last_t)
            self._last_t = t
            self._pbar.close()
            self._pbar = None

    def on_run_error(self, step, error, driver):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


class ProgressBarCallback(AbstractCallback, mutable=True):
    _leave: bool = struct.field(pytree_node=False, serialize=False)

    _n_steps: int = struct.field(pytree_node=False, serialize=False)
    _last_step: int = struct.field(pytree_node=False, serialize=False)

    _pbar: Any = struct.field(pytree_node=False, serialize=False)

    def __init__(self, n_steps: int, leave: bool = True):
        self._pbar = None
        self._n_steps = n_steps
        self._leave = leave

    def on_run_start(self, step, driver):
        self._last_step = driver.step_count

        self._pbar = tqdm(
            total=self._n_steps,
            disable=not jax.process_index() == 0,
            dynamic_ncols=True,
            leave=self._leave,
        )
        self._pbar.update(driver.step_count - self._last_step)
        self._pbar.unpause()
        self._pbar.refresh()

    def before_parameter_update(self, step, log_data, driver):
        # if the cost-function is defined then report it in the progress bar
        if driver._loss_stats is not None:
            self._pbar.set_postfix_str(
                driver._loss_name + "=" + str(driver._loss_stats)
            )

    def on_step_end(self, step, log_data, driver):
        step_value = driver.step_count - self._last_step
        self._pbar.update(step_value - self._pbar.n)
        self._pbar.refresh()

    def on_run_end(self, step, driver):
        step_value = driver.step_count - self._last_step
        if self._pbar is not None:
            self._pbar.update(step_value - self._pbar.n)
            self._pbar.refresh()
            self._pbar.close()
            self._pbar = None

    def on_run_error(self, step, error, driver):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
