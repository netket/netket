from netket.utils import struct


class StopRun(Exception):
    """Exception to be raised by callbacks to stop the run of the driver."""

    pass


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

    def on_run_start(self, step, driver):
        pass

    def on_step_start(self, step, log_data, driver):
        pass

    def on_compute_update_start(self, step, log_data, driver):
        pass

    def on_compute_update_end(self, step, log_data, driver) -> bool:
        """
        Callback called at the end of the compute update phase, after computing the loss and its gradient.

        This is called before the parameters are updated, so it can be used to implement custom
        logic for rejecting a step based on the computed loss or gradient.

        Returns:
            A boolean indicating whether to reject the step (i.e. repeat it with the same parameters).
            If it returns None, it is treated as False.
        """
        return False

    def on_legacy_run(self, step, log_data, driver):
        pass

    def on_parameter_update(self, step, log_data, driver):
        pass

    def on_step_end(self, step, log_data, driver):
        pass

    def on_run_end(self, step, driver):
        pass

    def on_run_error(self, step, error, driver):
        pass
