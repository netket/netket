from netket.utils import struct


class StopRun(Exception):
    """
    Exception to be raised by callbacks to gracefully stop the optimisation loop.

    Raise this exception (or a subclass of it) from any callback hook to stop
    the :meth:`~netket.driver.AbstractDriver.run` loop early.
    The driver will catch it, call
    :meth:`~netket.callbacks.AbstractCallback.on_run_end` on all callbacks, and
    then return normally — no traceback is printed and no exception propagates to
    the caller.

    The message passed to the exception will be printed to stdout.

    Example::

        class StopAfterConvergence(nk.callbacks.AbstractCallback):
            def on_step_end(self, step, log_data, driver):
                if log_data["Energy"].error_of_mean < 1e-4:
                    raise nk.callbacks.StopRun("Energy converged.")
    """

    pass


#: ``callback_order`` for callbacks that may raise :class:`StopRun`. They run
#: *last* (after observables at 0 and loggers at 10) so that, on the step a
#: callback stops the run, no later callback in the hook is skipped — a skipped
#: collective would deadlock the run under ``jax.distributed``.
STOPPING_CALLBACK_ORDER = 100


class AbstractCallback(struct.Pytree, mutable=True):
    """
    Abstract base class for callbacks in advanced variational drivers.

    This class is a Pytree, so it can be used with JAX transformations and
    automatically handles serialisation, but fields must be declared
    with `struct.field(pytree_node=False)` as class attributes.

    Subclass this class and override any of the hook methods to inject custom
    logic at specific points of the optimisation loop.  All hook methods have
    no-op default implementations, so you only need to override the ones you
    need.

    To stop the optimisation early from inside any hook, raise
    :class:`~netket.callbacks.StopRun` (or a subclass of it).  The driver will
    catch it, call :meth:`on_run_end` on all callbacks, and return normally.

    For a full description of the run loop structure and every available hook,
    including pseudocode showing exactly when each hook is called, see
    :ref:`advanced_custom_callbacks`.
    """

    def __init_subclass__(cls, **kwargs):
        from netket.errors import CallbackLegacyHookError

        super().__init_subclass__(**kwargs)

        old_methods = [
            m for m in ("on_legacy_run", "on_parameter_update") if m in cls.__dict__
        ]
        if old_methods:
            raise CallbackLegacyHookError(cls.__name__, old_methods)

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
        (Default: 0, for all callbacks, 10 for loggers, and
        :data:`STOPPING_CALLBACK_ORDER` (100) for callbacks that may raise
        :class:`StopRun`, so they run after everything else).
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

    def before_parameter_update(self, step, log_data, driver):
        """
        Called after all update logic has been computed and the step has been accepted,
        but before the driver applies the parameter update.

        At this point:

        - The loss and its gradient have been computed by :meth:`~netket.driver.AbstractDriver.compute_loss_and_update`.
        - The step has been accepted (not rejected by :meth:`on_compute_update_end`).
        - ``driver.step_count`` still refers to the *current* step — it has not yet been incremented.
        - The variational state parameters have **not** yet changed.

        This is the right place to estimate additional observables, add data to
        ``log_data``, or take a snapshot of the state for logging.  Callbacks with a
        lower :attr:`callback_order` run first, so observables callbacks (order 0) are
        guaranteed to populate ``log_data`` before logger callbacks (order 10) read it.
        """
        pass

    def on_step_end(self, step, log_data, driver):
        pass

    def on_run_end(self, step, driver):
        pass

    def on_run_error(self, step, error, driver):
        pass

    def on_checkpoint_save(self, step, driver):
        """Called by a checkpointer just before a checkpoint is committed at ``step``.

        Override to force this callback's **external durable state** (e.g. a
        logger's output file) up to ``step`` *before* the checkpoint commits, so
        that store never lags the checkpoint. Default: no-op.

        This is part of the checkpointing participation protocol; it is *not*
        called by the normal run loop, only by a checkpointing callback (see
        :mod:`netket.checkpoint`).
        """
        pass

    def on_checkpoint_restore(self, step, driver):
        """Called by a checkpointer after the driver has been restored to ``step``.

        Override to reconcile this callback's **external durable state** to a run
        resuming at ``step``, *after* the serialized fields have been restored —
        e.g. a logger reloading its file and dropping every entry at iteration
        ``>= step`` so it does not run ahead of the checkpoint. Default: no-op.
        """
        pass
