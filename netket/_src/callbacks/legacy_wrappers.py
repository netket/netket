from typing import Any
from functools import wraps
import copy

from netket.vqs import VariationalState
from netket.utils import struct

from advanced_drivers._src.callbacks.base import AbstractCallback


class LegacyCallbackWrapper(AbstractCallback):
    callback: Any

    def __init__(self, callback: Any):
        self.callback = callback

    def on_legacy_run(self, step, log_data, driver):
        do_continue = self.callback(step, log_data, driver)
        if not do_continue:
            driver._stop_run = True


class LegacyLoggerWrapper(AbstractCallback):
    logger: Any

    _vstate: VariationalState | None = struct.field(serialize=False)
    """
    Used to store the variational state between when parameters are updated and
    when we actually log it.
    """

    def __init__(self, logger: Any):
        self.logger = logger
        self._vstate = None

    @property
    @wraps(AbstractCallback.callback_order)
    def callback_order(self) -> int:
        return 10

    def on_parameter_update(self, step, log_data, driver):
        # Store the vstate internally so that if we modify the parameters
        # we can still log the previous one.
        self._vstate = copy.copy(driver.state)

    def on_step_end(self, step, log_data, driver):
        self.logger(step, log_data, self._vstate)

    def on_run_end(self, step, driver):
        self.logger.flush(driver.state)
        self._vstate = None

    def on_run_error(self, step, error, driver):
        self._vstate = None
