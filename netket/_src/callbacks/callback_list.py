from typing import Any


from advanced_drivers._src.callbacks.base import AbstractCallback


class CallbackList(AbstractCallback):
    callbacks: tuple[Any]

    def __init__(
        self,
        callbacks=None,
        # add_history=False,
        # add_progbar=False,
        # model=None,
        # **params,
    ):
        if isinstance(callbacks, AbstractCallback):
            callbacks = (callbacks,)
        self.callbacks = tuple(sorted(callbacks, key=lambda x: x.callback_order))

    def on_run_start(self, step, driver, callbacks):
        for callback in self.callbacks:
            callback.on_run_start(step, driver, callbacks)

    def on_iter_start(self, step, log_data, driver):
        for callback in self.callbacks:
            callback.on_iter_start(step, log_data, driver)

    def on_step_start(self, step, log_data, driver):
        for callback in self.callbacks:
            callback.on_step_start(step, log_data, driver)

    def on_reset_step_end(self, step, log_data, driver):
        for callback in self.callbacks:
            callback.on_reset_step_end(step, log_data, driver)

    def on_compute_update_start(self, step, log_data, driver):
        for callback in self.callbacks:
            callback.on_compute_update_start(step, log_data, driver)

    def on_compute_update_end(self, step, log_data, driver):
        for callback in self.callbacks:
            callback.on_compute_update_end(step, log_data, driver)

    def on_parameter_update(self, step, log_data, driver):
        for callback in self.callbacks:
            callback.on_parameter_update(step, log_data, driver)

    def on_step_end(self, step, log_data, driver):
        for callback in self.callbacks:
            callback.on_step_end(step, log_data, driver)

    def on_legacy_run(self, step, log_data, driver):
        for callback in self.callbacks:
            callback.on_legacy_run(step, log_data, driver)

    def on_iter_end(self, step, log_data, driver):
        for callback in self.callbacks:
            callback.on_iter_end(step, log_data, driver)

    def on_run_end(self, step, driver):
        for callback in self.callbacks:
            callback.on_run_end(step, driver)

    def on_run_error(self, step, error, driver):
        for callback in self.callbacks:
            callback.on_run_error(step, error, driver)
