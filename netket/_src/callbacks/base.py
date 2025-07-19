from netket.utils import struct


class AbstractCallback(struct.Pytree, mutable=True):
    """
    Abstract base class for callbacks in advanced variational drivers.

    This class is a Pytree, so it can be used with JAX transformations and
    automatically handles serialisation, but fields must be declared
    with `struct.field(pytree_node=False)` as class attributes.
    """

    def __init__(
        self,
    ):
        pass

    @property
    def callback_order(self) -> int:
        """
        An integer representing the order in which this callback should be called.

        Lower numbers are called first, and higher numbers are called later.

        This can be redefined in subclasses to change the order in which callbacks are called.
        (Default: 0, for all callbacks, 10 for loggers).
        """
        return 0

    def on_run_start(self, step, driver, callbacks):
        pass

    def on_reset_step_end(self, step, driver, callbacks):
        pass

    def on_iter_start(self, step, log_data, driver):
        pass

    def on_step_start(self, step, log_data, driver):
        pass

    def on_compute_update_start(self, step, log_data, driver):
        pass

    def on_compute_update_end(self, step, log_data, driver):
        pass

    def on_parameter_update(self, step, log_data, driver):
        pass

    def on_step_end(self, step, log_data, driver):
        pass

    def on_legacy_run(self, step, log_data, driver):
        pass

    def on_iter_end(self, step, log_data, driver):
        pass

    def on_run_end(self, step, driver):
        pass

    def on_run_error(self, step, error, driver):
        pass
