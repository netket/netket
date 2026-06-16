from netket.utils import struct


class StopRun(Exception):
    """
    A request to stop the :meth:`~netket.driver.AbstractDriver.run` loop early,
    returned by :meth:`~netket.callbacks.AbstractCallback.on_step_end`. The run
    stops after that step, on every process, calls ``on_run_end`` on all
    callbacks and returns normally; the message is printed.

    Raising it from ``on_step_end`` also works. Raised from any other hook, it is
    an error: the run fails with a ``RuntimeError``.

    Example::

        class StopAfterConvergence(nk.callbacks.AbstractCallback):
            def on_step_end(self, step, log_data, driver):
                if log_data["Energy"].error_of_mean < 1e-4:
                    return nk.callbacks.StopRun("Energy converged.")
    """

    pass


class RunFinished(StopRun):
    """
    A :class:`~netket.callbacks.StopRun` for a run that has reached its end, for
    example the final time of a time evolution.

    It works like :class:`~netket.callbacks.StopRun`, but the driver does not
    print that the run is stopping early, and the run counts as completed. The
    step that returns it is not counted in ``step_count``: it is meant for a step
    that only reports the end of the run, such as the last step of a time
    evolution, which logs the final time.
    """


#: ``callback_order`` for callbacks that may stop the run. They run *last*
#: (after observables at 0 and loggers at 10), on a step's complete log.
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

    To stop the optimisation early, return a :class:`~netket.callbacks.StopRun`
    from :meth:`on_step_end`.

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

    def on_run_prep_allocate(self, driver):
        """Called first in every :meth:`~netket.driver.AbstractDriver.run`, before
        a checkpoint is restored. Allocate here the saved state whose shape
        depends on the driver (e.g. a buffer shaped like the parameters), so that
        a checkpoint can be restored into it. A cheap initialisation needed only
        on a fresh start can go here too: a restore overwrites it. Nothing else
        (no logging, no files): the step is not restored yet. Default: no-op.
        """
        pass

    def on_run_start(self, step, driver):
        """Called when a run starts at step ``step``, after any restore. A
        resumed run calls it again, so it must not change what the run computes
        (e.g. by sampling); setting up files or tuning chunk sizes is fine.
        Loggers drop here what an earlier run logged at iteration ``>= step``.

        An expensive initialisation needed only on a fresh start (e.g. a long
        thermalisation) can run here if :meth:`on_checkpoint_restore` was not
        called since :meth:`on_run_prep_allocate`.
        """
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
        """Called at the end of an accepted step ``step``, after the parameter
        update. Returns ``None``, or a :class:`StopRun` to stop the run after this
        step."""
        pass

    def on_run_end(self, step, driver):
        """Called once when a run ends, normally or stopped (not when it fails:
        see :meth:`on_run_error`)."""
        pass

    def on_run_error(self, step, error, driver):
        pass

    def on_checkpoint_save(self, step, driver):
        """Called before the checkpointer saves the run at step ``step``. Write
        here what this callback keeps outside the driver (e.g. a log file) up to
        ``step``, so it never lags the checkpoint. It must not change what the
        run computes. Default: no-op.
        """
        pass

    def on_checkpoint_restore(self, step, driver):
        """Called when the run was restored from a checkpoint of step ``step``,
        before :meth:`on_run_start`; not when there was nothing to restore. What
        every run start needs belongs in :meth:`on_run_start`. Default: no-op.
        """
        pass
