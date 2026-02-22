from typing import Any

from tqdm.auto import tqdm

import jax

from netket.utils import struct

from netket._src.callbacks.base import AbstractCallback


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

    def on_legacy_run(self, step, log_data, driver):
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
