# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Union
from collections.abc import Iterable
from pathlib import Path
import sys
import time

import numbers

import jax

from netket.logging import AbstractLog, JsonLog
from netket.operator._abstract_observable import AbstractObservable
from netket.utils import struct, timing
from netket.utils.iterators import to_iterable
from netket.utils.types import PyTree
from netket.vqs import VariationalState, FullSumState

from netket._src import distributed
from netket._src.callbacks.base import AbstractCallback, StopRun, RunFinished
from netket._src.callbacks.observables import ObservableCallback
from netket._src.callbacks.legacy_wrappers import (
    LegacyCallbackT,
    LegacyCallbackWrapper,
    LegacyLoggerWrapper,
)
from netket._src.callbacks.progressbar import ProgressBarCallback
from netket._src.callbacks.callback_list import CallbackList
from netket._src.driver.checkpointer import (
    AbstractCheckpointer,
    checkpoint_state,
    load_checkpoint_state,
)


def maybe_wrap_legacy_callback(callback):
    if isinstance(callback, AbstractCallback):
        return callback
    else:
        return LegacyCallbackWrapper(callback)


class AbstractDriver(struct.Pytree, mutable=True):
    """Abstract base class for NetKet variational drivers.

    This class must be inherited from in order to create a driver that
    immediately works with NetKet loggers and callback mechanism.

    The :meth:`run` method executes the driver loop and invokes
    :class:`~netket.callbacks.AbstractCallback` hooks at well-defined points during each step.
    To stop the loop early from a callback, return :class:`~netket.callbacks.StopRun`.
    For a detailed description of the loop structure and the available callback hooks, see
    :ref:`advanced_custom_callbacks`.

    .. note::

        How to implement a new driver

        For a concrete example, look at the file `netket/driver/vmc.py`.

        To implement a new **optimization** driver (with an optax optimizer), subclass
        :class:`~netket.driver.AbstractOptimizationDriver` instead.

        To implement a new **dynamics** driver (time evolution), subclass
        :class:`~netket.driver.AbstractDynamicsDriver` instead.

        If subclassing :class:`AbstractDriver` directly, define:

        - :meth:`~netket.driver.AbstractDriver.__init__`: call ``super().__init__(variational_state)``
          and optionally pass ``minimized_quantity_name``.

        - :meth:`~netket.driver.AbstractDriver.compute_loss_and_update`:
          compute the loss and the parameter update, returning both as a tuple.

        - :meth:`~netket.driver.AbstractDriver.update_parameters`:
          apply the parameter update returned by :meth:`compute_loss_and_update`.

        - :meth:`~netket.driver.AbstractDriver.reset_step` (optional):
          reset the sampler at the start of each step.

        To support checkpointing, set every saved field in ``__init__``, with the
        same structure, shapes and dtypes it will have after running, instead of
        creating it at the first step. A checkpoint can then be restored into a
        newly created driver. Values recomputed at every step should not be saved
        (``serialize=False``, see :func:`netket.utils.struct.field`).

    """

    # Configuration fields, not very important
    _loss_name: str = struct.field(pytree_node=False, serialize=False)

    # Stuff to iterate the driver
    _loss_stats: Any = struct.field(serialize=False, default=None)
    _step_count: int = struct.field(serialize=True)

    # state of the driver
    _variational_state: VariationalState = struct.field(
        pytree_node=False,
        serialize=True,
        serialize_name="state",
    )

    # Persistent (behavioural) callbacks declared at construction. Stored the same
    # way as `_variational_state` (static aux data, but serialized) so that their
    # state round-trips inside the driver's serialized form. See `callbacks`.
    _callbacks: tuple = struct.field(pytree_node=False, serialize=True, default=())

    # Internal caches (those could be removed in the future?)
    _dp: PyTree = struct.field(pytree_node=True, serialize=False)

    # The steps at which the current (or last) call to `run` started and stops.
    _start_step: int | None = struct.field(
        pytree_node=False, serialize=False, default=None
    )
    _target_step: int | None = struct.field(
        pytree_node=False, serialize=False, default=None
    )
    _step_attempt: int = struct.field(pytree_node=False, serialize=False, default=0)
    _timer: timing.Timer = struct.field(pytree_node=False, serialize=False)

    def __init__(
        self,
        variational_state: VariationalState,
        minimized_quantity_name: str = "loss",
        *,
        callbacks: Iterable[AbstractCallback] = (),
    ):
        """
        Initializes a variational driver.

        Args:
            variational_state: The variational state.
            minimized_quantity_name: the name of the loss function in
                the logged data set.
            callbacks: An iterable of persistent (behavioural) callbacks attached
                to the driver itself, as opposed to the per-run callbacks passed
                to :meth:`run`. These are merged with the per-run callbacks every
                time :meth:`run` is called, and — being part of the driver's
                serialized state — their internal state survives a
                checkpoint/restore round-trip. They must be
                :class:`~netket.callbacks.AbstractCallback` instances whose
                serialized fields can be saved, and are fixed for the lifetime
                of the driver.
        """
        self._loss_stats = None
        self._loss_name = minimized_quantity_name
        self._step_count = 0

        self._variational_state = variational_state
        callbacks = tuple(to_iterable(callbacks, none_is_empty=True))
        for callback in callbacks:
            if isinstance(callback, AbstractCheckpointer):
                raise TypeError("Pass a checkpointer as run(checkpointer=...).")
            if not isinstance(callback, AbstractCallback) or isinstance(
                callback, (LegacyCallbackWrapper, LegacyLoggerWrapper)
            ):
                raise TypeError(
                    "A persistent callback must be an AbstractCallback whose state "
                    f"can be saved, not a {type(callback).__name__}: pass functions "
                    "and loggers to run() instead."
                )
        self._callbacks = callbacks

        self._dp = jax.tree.map(jax.numpy.zeros_like, self.state.parameters)

    def _forward_and_backward(self):
        self.reset_step()
        self._loss_stats, self._dp = self.compute_loss_and_update()
        return self._loss_stats, self._dp

    def __init_subclass__(cls, **kwargs):
        import inspect
        import warnings
        from netket.errors import (
            ForwardAndBackwardDeprecationWarning,
            LogAdditionalDataSignatureDeprecationWarning,
        )

        super().__init_subclass__(**kwargs)

        # Backward compatibility: _forward_and_backward was renamed to
        # compute_loss_and_update. The signatures also differ:
        #   old: _forward_and_backward(self) -> gradient        (sets self._loss_stats as side-effect)
        #   new: compute_loss_and_update(self) -> (loss_stats, gradient)
        if (
            "_forward_and_backward" in cls.__dict__
            and "compute_loss_and_update" not in cls.__dict__
        ):
            warnings.warn(
                ForwardAndBackwardDeprecationWarning(cls.__name__), stacklevel=2
            )

            old_fab = cls.__dict__["_forward_and_backward"]

            def compute_loss_and_update(self, _old_fab=old_fab):
                dp = _old_fab(self)
                return self._loss_stats, dp

            cls.compute_loss_and_update = compute_loss_and_update

        # Backward compatibility: _log_additional_data used to receive an explicit
        # `step` argument; it has since been removed (use self.step_count instead).
        if "_log_additional_data" in cls.__dict__:
            old_lad = cls.__dict__["_log_additional_data"]
            params = list(inspect.signature(old_lad).parameters)
            # inspect on an unbound function includes 'self'; strip it to count
            # only the domain-specific parameters.
            non_self = [p for p in params if p != "self"]
            if len(non_self) == 2:  # old signature: (log_dict, step)
                warnings.warn(
                    LogAdditionalDataSignatureDeprecationWarning(cls.__name__),
                    stacklevel=2,
                )

                def _log_additional_data(self, log_dict, _old_lad=old_lad):
                    _old_lad(self, log_dict, self.step_count)

                cls._log_additional_data = _log_additional_data

    def compute_loss_and_update(self) -> PyTree:  # pragma: no cover
        """
        :meta public:

        Performs a step of the optimization driver, returning the PyTree
        of the gradients that will be optimized.

        Concrete drivers must override this method.

        .. note::

            When implementing this function on a subclass, you must return the
            gradient which must match the pytree structure of the parameters
            of the variational state.

            The gradient will then be passed on to the optimizer in order to update
            the parameters.

            Moreover, if you are minimising a loss function you must set the
            field `self._loss_stats` with the current value of the loss function.

            This will be logged to any logger during optimisation.

        Returns:
            the update for the weights.
        """
        raise NotImplementedError()  # pragma: no cover

    def reset(self):
        """
        .. deprecated:: 3.22

            Use :meth:`reset_step` to reset the sampler state at the beginning of a step.
            Note that the old ``reset()`` also reset ``step_count`` to 0; this behaviour
            is no longer supported.
        """
        import warnings
        from netket.errors import DriverResetDeprecationWarning

        warnings.warn(DriverResetDeprecationWarning(), stacklevel=2)
        self.reset_step()
        # self._step_count = 0

    def _accept_step(self) -> bool:
        """
        Called after :meth:`compute_loss_and_update` to decide whether the
        current step should be accepted.

        Return ``False`` to reject the step and retry (``_step_attempt`` will
        be incremented before the next attempt). Subclasses can override this
        to implement driver-level acceptance criteria such as adaptive
        step-size control: mutate driver state (e.g. ``self.dt``) as needed
        before returning ``False``.

        The default implementation always accepts.
        """
        return True

    def reset_step(self, hard: bool = False):
        """
        Resets the state of the driver at the beginning of a new step.

        This method is called at the beginning of every step in the optimization.

        Args:
            hard: If True, the reset is a hard reset, resulting in a complete resampling even if `resample_fraction`
            is not `None`.
        """
        if hard:
            self.state.reset_hard()
        else:
            self.state.reset()

    def _log_additional_data(self, log_dict: dict):
        """
        Method to be implemented in sub-classes of AbstractDriver to
        log additional data at every step.
        This method is called at every iteration when executing with `run`.

        Args:
            log_dict: The dictionary containing all logged data. It must be
                **modified in-place** adding new keys.
            step: the current step number.
        """
        # Always log the acceptance.
        if hasattr(self.state, "sampler_state"):
            acceptance = getattr(self.state.sampler_state, "acceptance", None)
            if acceptance is not None:
                log_dict["acceptance"] = acceptance

        log_dict["wallclock"] = time.perf_counter()

    @property
    def state(self):
        """
        Returns the machine that is optimized by this driver.
        """
        return self._variational_state

    @property
    def step_count(self):
        """
        Returns a monotonic integer labelling all the steps performed by this driver.
        This can be used, for example, to identify the line in a log file.
        """
        return self._step_count

    @property
    def callbacks(self) -> tuple[AbstractCallback, ...]:
        """
        The tuple of persistent (behavioural) callbacks given to the constructor,
        fixed for the lifetime of the driver.

        These callbacks are part of the driver's (serialized) state and are merged
        with the per-run callbacks every time :meth:`run` is called. The per-run
        callbacks are not saved.
        """
        return self._callbacks

    def _default_callbacks(
        self,
        callbacks,
        *,
        n_iter: int,
        obs,
        step_size,
        loggers,
        show_progress,
        **kwargs,
    ) -> CallbackList:
        """
        Function used to create the default callbacks for the run method, given some user inputs.

        This function can be overridden by subclasses to create custom default callbacks, for example
        to add a custom progress bar, or to automatically log more things.

        Example of subclassing:

        .. code:: python
            class MyDriver(AbstractDriver):
                ...
                def _default_callbacks(self, *args, **kwargs):
                    callbacks = super()._default_callbacks(*args, **kwargs)
                    callbacks.callbacks.append(MyCustomCallback())
                    return callbacks

        Args:
            callbacks: the user-provided callbacks, which can be None, a single callback or an iterable of callbacks.
            n_iter: the number of iterations to be performed during the run, used to create the progress bar callback.
            obs: the observables to be logged, used to create the observable callback.
            step_size: the frequency with which observables should be logged, used to create the observable callback.
            loggers: the loggers to be used during the run, used to create the legacy logger callbacks.
            show_progress: whether to show the progress bar, used to decide whether to create the progress bar callback.

        Returns:
            A CallbackList containing all the callbacks to be used during the run.
        """
        callback_list = [maybe_wrap_legacy_callback(c) for c in callbacks]
        if obs is not None:
            callback_list.append(ObservableCallback(obs, step_size))
        for log in loggers:
            if isinstance(log, AbstractCallback):
                callback_list.append(log)
            elif isinstance(log, AbstractLog):
                callback_list.append(LegacyLoggerWrapper(log))
            else:
                callback_list.append(LegacyLoggerWrapper(log))
        if show_progress:
            callback_list.append(ProgressBarCallback(n_iter))
        callbacks = CallbackList(callback_list)
        return callbacks

    def run(
        self,
        n_iter: int,
        out: Iterable[AbstractLog] | None = (),
        obs: dict[str, AbstractObservable] | None = None,
        step_size: int = 1,
        show_progress: bool = True,
        save_params_every: int = 50,  # for default logger
        write_every: int = 50,  # for default logger
        callback: Union[LegacyCallbackT, AbstractCallback, None] = None,
        timeit: bool = False,
        checkpointer: AbstractCheckpointer | None = None,
        _graceful_keyboard_interrupt: bool = True,
    ):
        """
        Runs this variational driver, updating the weights of the network stored in
        this driver for `n_iter` steps and dumping values of the observables `obs`
        in the output `logger`.

        It is possible to control more specifically what quantities are logged, when to
        stop the optimisation, or to execute arbitrary code at every step by specifying
        one or more callbacks, which are passed as a list of functions to the keyword
        argument `callback`.

        Callbacks are functions that follow this signature:

        .. Code::

            def callback(step, log_data, driver) -> bool:
                ...
                return True/False

        If a callback returns True, the optimisation continues, otherwise it is stopped.
        The `log_data` is a dictionary that can be modified in-place to change what is
        logged at every step. For example, this can be used to log additional quantities
        such as the acceptance rate of a sampler.

        Alternatively, :class:`~netket.callbacks.AbstractCallback` subclasses can be
        used to hook into more stages of the loop.  To stop the optimisation early,
        return a :class:`~netket.callbacks.StopRun` from ``on_step_end``: the run
        stops after that step, counted in :attr:`step_count`, all callbacks are
        finalised via their ``on_run_end`` method, and ``run`` returns normally.

        Loggers are specified as an iterable passed to the keyword argument `out`. If only
        a string is specified, this will create by default a :class:`nk.logging.JsonLog`.
        To know about the output format check its documentation. The logger object is
        also returned at the end of this function so that you can inspect the results
        without reading the json output.

        With a ``checkpointer`` (experimental), the run resumes from its last
        checkpoint:

        1. every callback's ``on_run_prep_allocate`` (to allocate the state it
           saves), ``checkpointer.restore``, ``on_checkpoint_restore`` if a
           checkpoint was restored, then ``on_run_start``;
        2. after each step, ``checkpointer.should_stop``, then, if
           ``checkpointer.should_save``, every callback's ``on_checkpoint_save``
           and ``checkpointer.save``;
        3. ``checkpointer.wait`` before ``on_run_end``, and ``checkpointer.close``
           with what ended the run (``None`` if it completed). A failure to close
           is raised, chained to the run's error if any.

        A run stops only at the end of a step, on every process, when a callback
        (``on_step_end``) or the checkpointer asks: a resumed run whose stopper is
        already satisfied takes one step, then stops. A stopped run can be
        resumed. Nothing is saved before the first step: a fresh run killed
        before its first save starts over. Code that rejects steps must raise an
        error if it can never accept one.

        Args:
            n_iter: the number of steps to run, counted from the step count when
                `run` is called.
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            obs: An iterable containing all observables that should be computed
            step_size: Every how many steps should observables be logged to disk (default=1)
            callback: Callable or list of callable callback functions to stop training given a condition
            show_progress: If true displays a progress bar (default=True)
            save_params_every: Every how many steps the parameters of the network should be
                serialized to disk (ignored if logger is provided)
            write_every: Every how many steps the json data should be flushed to disk (ignored if
                logger is provided)
            timeit: If True, provide timing information.
            checkpointer: A checkpointer (experimental: see
                :class:`~netket.experimental.checkpoint.AbstractCheckpointer`), to
                save the run and resume it from its last checkpoint (see above).
                With one, a logger created from a string `out` appends to its file.
            _graceful_keyboard_interrupt: (Internal flag, defaults to True) If True, the driver will gracefully
                handle a KeyboardInterrupt, usually arising from doing ctrl-C, returning the current state of the
                simulation. If False, the KeyboardInterrupt will be raised as usual.
                This only has an effect when running in interactive mode.
        """

        if not isinstance(n_iter, numbers.Number):
            raise ValueError(
                "n_iter, the first positional argument to `run`, must be a number!"
            )

        if not isinstance(checkpointer, AbstractCheckpointer | None):
            raise TypeError(
                "checkpointer must be one AbstractCheckpointer, not a "
                f"{type(checkpointer).__name__}."
            )

        # A path `out` gets a JsonLog: appended to with a checkpointer (it drops the
        # rows from the start step on), overwritten otherwise.
        if isinstance(out, (str, Path)):
            mode = "w" if checkpointer is None else "a"
            out = (JsonLog(out, mode, save_params_every, write_every),)
        elif out is None:
            out = ()
        loggers = to_iterable(out)

        # Merge the persistent (behavioural) callbacks declared on the driver with
        # the per-run ones. Persistent callbacks run first so that a per-run
        # `callback=` keeps appending semantics; ordering between callbacks is
        # ultimately decided by `callback_order` inside the CallbackList.
        run_callbacks = (
            *self._callbacks,
            *to_iterable(callback, none_is_empty=True),
        )
        if any(isinstance(c, AbstractCheckpointer) for c in run_callbacks):
            raise TypeError("Pass a checkpointer as run(checkpointer=...).")
        callbacks = self._default_callbacks(
            run_callbacks,
            n_iter=n_iter,
            obs=obs,
            step_size=step_size,
            loggers=loggers,
            show_progress=show_progress,
        )

        # Fixed before the restore and `on_run_start`, so that restoring
        # `step_count` (when resuming from a checkpoint) does not move the end of
        # the run.
        self._start_step = self.step_count
        self._target_step = self.step_count + n_iter

        cause = None
        with timing.timed_scope(force=timeit) as timer:
            try:
                try:
                    # Allocate, then restore the run, before any other callback hook.
                    callbacks.on_run_prep_allocate(self)
                    if checkpointer is not None:
                        state = checkpointer.restore(
                            self,
                            start_step=self._start_step,
                            target_step=self._target_step,
                        )
                        if state is not None:
                            load_checkpoint_state(self, state)
                            callbacks.on_checkpoint_restore(self.step_count, self)
                    callbacks.on_run_start(self.step_count, self)
                    stop = None
                    while stop is None and self.step_count < self._target_step:
                        self._step_attempt = 0
                        step_log_data = {}

                        while True:
                            callbacks.on_step_start(
                                self.step_count, step_log_data, self
                            )

                            self.reset_step()

                            callbacks.on_compute_update_start(
                                self.step_count, step_log_data, self
                            )
                            self._loss_stats, self._dp = self.compute_loss_and_update()
                            reject_step = callbacks.on_compute_update_end(
                                self.step_count, step_log_data, self
                            )
                            driver_reject_step = not self._accept_step()
                            if driver_reject_step or reject_step:
                                self._step_attempt += 1
                                continue
                            else:
                                break
                        # If we are here, we accepted the step
                        if self._loss_stats is not None:
                            step_log_data[self._loss_name] = self._loss_stats

                        self._log_additional_data(step_log_data)
                        callbacks.before_parameter_update(
                            self.step_count, step_log_data, self
                        )
                        self.update_parameters(self._dp)

                        stop = callbacks.on_step_end(
                            self.step_count, step_log_data, self
                        )
                        # A step that ends the run with RunFinished only reports
                        # its end (e.g. the final time of a time evolution).
                        if not isinstance(stop, RunFinished):
                            self._step_count += 1
                        if checkpointer is not None:
                            # Asked on every process at every step.
                            request = checkpointer.should_stop(self.step_count)
                            stop = request if stop is None else stop
                        # One process's stop stops them all (asked first: every
                        # process must take part).
                        if distributed.any_rank(stop is not None) and stop is None:
                            stop = StopRun(
                                "Another process stopped the run at step "
                                f"{self.step_count}."
                            )
                        # The save point of the step, after every callback's
                        # `on_step_end` and before the next step.
                        self._maybe_save_checkpoint(
                            callbacks,
                            checkpointer,
                            last=stop is not None
                            or self.step_count >= self._target_step,
                        )

                    # A run that reached its end (e.g. the final time of a time
                    # evolution) completed: it did not stop early.
                    if isinstance(stop, RunFinished):
                        stop = None
                    cause = stop
                    # The checkpoints are written before the callbacks hear the end.
                    if checkpointer is not None:
                        checkpointer.wait()
                    callbacks.on_run_end(self.step_count, self)
                except StopRun as error:
                    raise RuntimeError(
                        "StopRun may only be returned (or raised) from on_step_end, "
                        f"not raised elsewhere: {error}"
                    ) from error
                if stop is not None:
                    print(
                        f"[process {jax.process_index()}/{jax.process_count()}] "
                        f"Stopping early because of: {stop}"
                    )
            except KeyboardInterrupt as error:
                cause = error
                callbacks.on_run_error(self.step_count, error, self)
                if _graceful_keyboard_interrupt and hasattr(sys, "ps1"):
                    print("Stopped by user.")
                else:
                    raise
            except Exception as error:
                cause = error
                callbacks.on_run_error(self.step_count, error, self)
                raise
            finally:
                # A close failure raised here keeps the run's error as __context__.
                if checkpointer is not None:
                    checkpointer.close(cause)

        if timeit:
            self._timer = timer
            if jax.process_index() == 0:
                print(timer)

        return loggers

    def _maybe_save_checkpoint(self, callbacks, checkpointer, last):
        """
        Saves a checkpoint of step ``step_count`` if ``checkpointer`` asks for one
        (``last`` when the run ends there): every callback writes out what it keeps
        outside the driver (``on_checkpoint_save``), then the checkpointer saves the
        driver.
        """
        if checkpointer is None or not checkpointer.should_save(
            self.step_count, last=last
        ):
            return
        callbacks.on_checkpoint_save(self.step_count, self)
        checkpointer.save(self.step_count, checkpoint_state(self))

    def estimate(self, observables, fullsum: bool = False):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.

        Args:
            observables: A pytree of operators for which statistics should be computed.

        Returns:
            A pytree of the same structure as the input, containing MCMC statistics
            for the corresponding operators as leaves.
        """

        # Do not unpack operators, even if they are pytrees!
        # this is necessary to support jax operators.
        vstate = self.state
        if fullsum:
            vstate = FullSumState(
                hilbert=vstate.hilbert,
                model=vstate.model,
                chunk_size=vstate.chunk_size,
                variables=vstate.variables,
            )
        return jax.tree_util.tree_map(
            vstate.expect,
            observables,
            is_leaf=lambda x: isinstance(x, AbstractObservable),
        )

    def update_parameters(self, dp):
        """
        Updates the parameters of the variational state.

        Subclasses must override this method.  Optimization drivers should
        apply the optimizer; dynamics drivers should apply the delta directly.

        Args:
            dp: the pytree containing the updates to the parameters.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement update_parameters. "
            "If you want an optimization driver, subclass AbstractOptimizationDriver."
        )
