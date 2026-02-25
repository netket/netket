from typing import TYPE_CHECKING, Any, Callable
from functools import wraps
import copy

from netket.vqs import VariationalState
from netket.utils import struct

from netket._src.callbacks.base import AbstractCallback, StopRun

if TYPE_CHECKING:
    from netket._src.driver.abstract_variational_driver import AbstractVariationalDriver

LegacyCallbackT = Callable[[int, dict, "AbstractVariationalDriver"], bool]


class LegacyCallbackWrapper(AbstractCallback):
    """
    Wraps a legacy callback function (i.e. a function that takes the step, log_data and driver as arguments)
    into a callback that can be used with the new callback system.

    The legacy callback should return a boolean indicating whether to continue the run (True) or stop it (False).
    """

    callback: LegacyCallbackT

    def __init__(self, callback: LegacyCallbackT):
        self.callback = callback

    def before_parameter_update(self, step, log_data, driver):
        do_continue = self.callback(step, log_data, driver)
        if not do_continue:
            raise StopRun(f"Legacy callback {self.callback} requested to stop the run.")


class LegacyLoggerWrapper(AbstractCallback):
    """
    Wraps a legacy logger function (i.e. a function that takes the step, log_data and variational state as arguments)
    into a callback that can be used with the new callback system.
    """

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

    def before_parameter_update(self, step, log_data, driver):
        # Store the vstate internally so that if we modify the parameters
        # we can still log the previous one.
        self._vstate = copy.copy(driver.state)

    def on_step_end(self, step, log_data, driver):
        self.logger(step, log_data, self._vstate)
        self._vstate = None

    def on_run_end(self, step, driver):
        self.logger.flush(driver.state)
        self._vstate = None

    def on_run_error(self, step, error, driver):
        self.logger.flush(driver.state)
        self._vstate = None
