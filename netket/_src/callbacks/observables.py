from typing import Any

from netket.utils import struct, timing

from netket._src.callbacks.base import AbstractCallback


class ObservableCallback(AbstractCallback, mutable=True):
    _observables: Any = struct.field(pytree_node=False, serialize=False)
    _interval: int = struct.field(pytree_node=False, serialize=False, default=1)
    _fullsum: bool = struct.field(pytree_node=False, serialize=False, default=False)
    _last_step: float = struct.field(
        pytree_node=False, serialize=False, default=float("-inf")
    )

    def __init__(
        self,
        observables: list | dict | None,
        interval: float | int = 1,
        on_step: bool = False,
        fullsum: bool = False,
    ):
        r"""
        Callback to estimate observables.
        The observables are estimated every `interval` steps.

        Args:
            observables (list | dict | None): The observables to estimate.
            interval (float | int, optional): The interval at which to estimate the observables. Defaults to 1.
            fullsum (bool, optional): Whether to calculate the observables in fullsummation, avoiding noise from Monte Carlo sampling. Defaults to False.
        """
        if observables is None:
            observables = {}
        self._observables = observables
        self._interval = interval
        self._fullsum = fullsum

        self._last_step = float("-inf")

    def before_parameter_update(self, step, log_data, driver):
        if hasattr(driver, "t"):
            step = driver.t
        if step >= self._last_step + self._interval:
            with timing.timed_scope(name="observables"):
                log_data.update(
                    driver.estimate(self._observables, fullsum=self._fullsum)
                )
            self._last_step = step
