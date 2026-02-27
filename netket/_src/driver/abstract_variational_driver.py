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
import sys
import time

import numbers
from functools import partial

import jax

from netket.logging import AbstractLog, JsonLog
from netket.operator._abstract_observable import AbstractObservable
from netket.utils import struct, timing
from netket.utils.iterators import to_iterable
from netket.utils.types import Optimizer, PyTree
from netket.vqs import VariationalState, FullSumState

from netket._src.callbacks.base import AbstractCallback, StopRun
from netket._src.callbacks.observables import ObservableCallback
from netket._src.callbacks.legacy_wrappers import (
    LegacyCallbackT,
    LegacyCallbackWrapper,
    LegacyLoggerWrapper,
)
from netket._src.callbacks.progressbar import ProgressBarCallback
from netket._src.callbacks.callback_list import CallbackList


def maybe_wrap_legacy_callback(callback):
    if isinstance(callback, AbstractCallback):
        return callback
    else:
        return LegacyCallbackWrapper(callback)


class AbstractVariationalDriver(struct.Pytree, mutable=True):
    """Abstract base class for NetKet Variational Monte Carlo drivers

    This class must be inherited from in order to create an optimization driver that
    immediately works with NetKet loggers and callback mechanism.

    The :meth:`run` method executes the optimisation loop and invokes
    :class:`~netket.callbacks.AbstractCallback` hooks at well-defined points during each step.
    To stop the loop early from a callback, raise :class:`~netket.callbacks.StopRun`.
    For a detailed description of the loop structure and the available callback hooks, see
    :ref:`advanced_custom_callbacks`.

    .. note::

        How to implement a new driver

        For a concrete example, look at the file `netket/driver/vmc.py`.

        If you want to inherit the nice interface of :class:`netket.driver.AbstractVariationalDriver`,
        you should subclass it, and define the following methods:

        - The :meth:`~netket.driver.AbstractVariationalDriver.__init__` method should be called
          with the machine, optimizer and optionally the name of the loss minimised. If this
          driver is minimising a loss function and you want its name to show up automatically
          in the progress bar/output files you should pass the optional keyword argument.

        - :meth:`~netket.driver.AbstractVariationalDriver.compute_loss_and_update`,
          that should compute the loss function and the gradient, returning both as a tuple.
          If the driver is minimizing or maximising some loss function,
          this quantity should be returned as the first element of the tuple so it is
          automatically logged.

        - :meth:`~netket.driver.AbstractVariationalDriver._estimate_stats` should return
          the expectation value over the variational state of a single observable.

        - :meth:`~netket.driver.AbstractVariationalDriver.reset_step`,
          should reset the driver (usually the sampler). The basic implementation will call
          :meth:`~netket.vqs.VariationalState.reset`, but you are responsible for resetting
          extra fields in the driver itself.

    """

    # Configuration fields, not very important
    _loss_name: str = struct.field(pytree_node=False, serialize=False)

    # Stuff to iterate the driver
    _loss_stats: Any = struct.field(serialize=True)
    _step_count: int = struct.field(serialize=True)

    # state of the driver
    _optimizer: Optimizer = struct.field(pytree_node=False, serialize=False)
    _optimizer_state: Any = struct.field(pytree_node=True, serialize=True)
    _variational_state: VariationalState = struct.field(
        pytree_node=False,
        serialize=True,
        serialize_name="state",
    )

    # Internal caches (those could be removed in the future?)
    _dp: PyTree = struct.field(pytree_node=True, serialize=False)

    # Iterator caches
    # _step_start: int = struct.field(pytree_node=False, serialize=False, default=None)
    # _step_size: int = struct.field(pytree_node=False, serialize=False, default=1)
    # _step_end: int = struct.field(pytree_node=False, serialize=False, default=None)
    _step_attempt: int = struct.field(pytree_node=False, serialize=False, default=0)
    _timer: timing.Timer = struct.field(pytree_node=False, serialize=False)

    def __init__(
        self,
        variational_state: VariationalState,
        optimizer: Optimizer,
        minimized_quantity_name: str = "loss",
    ):
        """
        Initializes a variational optimization driver.

        Args:
            variational_state: The variational state to be optimized
            optimizer: an `optax <https://optax.readthedocs.io/en/latest/>`_ optimizer.
                If you do not want
                to use an optimizer, just pass a sgd optimizer with
                learning rate `-1`.
            minimized_quantity_name: the name of the loss function in
                the logged data set.
        """
        self._loss_stats = None
        self._loss_name = minimized_quantity_name
        self._step_count = 0

        self._variational_state = variational_state
        self.optimizer = optimizer

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
        Method to be implemented in sub-classes of AbstractVariationalDriver to
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
    def optimizer(self):
        """
        The optimizer used to update the parameters at every iteration.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        if optimizer is not None:
            self._optimizer_state = optimizer.init(self.state.parameters)

    @property
    def step_count(self):
        """
        Returns a monotonic integer labelling all the steps performed by this driver.
        This can be used, for example, to identify the line in a log file.
        """
        return self._step_count

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
            class MyDriver(AbstractVariationalDriver):
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
        used to hook into more stages of the loop.  To stop the optimisation early from
        any callback hook, raise :class:`~netket.callbacks.StopRun`: the driver will
        catch it, finalise all callbacks via their ``on_run_end`` method, and return
        normally without propagating the exception.

        Loggers are specified as an iterable passed to the keyword argument `out`. If only
        a string is specified, this will create by default a :class:`nk.logging.JsonLog`.
        To know about the output format check its documentation. The logger object is
        also returned at the end of this function so that you can inspect the results
        without reading the json output.

        Args:
            n_iter: the total number of iterations to be performed during this run.
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
            _graceful_keyboard_interrupt: (Internal flag, defaults to True) If True, the driver will gracefully
                handle a KeyboardInterrupt, usually arising from doing ctrl-C, returning the current state of the
                simulation. If False, the KeyboardInterrupt will be raised as usual.
                This only has an effect when running in interactive mode.
        """

        if not isinstance(n_iter, numbers.Number):
            raise ValueError(
                "n_iter, the first positional argument to `run`, must be a number!"
            )

        # if out is a path, create an overwriting Json Log for output
        if isinstance(out, str):
            out = (JsonLog(out, "w", save_params_every, write_every),)
        elif out is None:
            out = ()
        loggers = to_iterable(out)

        callbacks = self._default_callbacks(
            to_iterable(callback, none_is_empty=True),
            n_iter=n_iter,
            obs=obs,
            step_size=step_size,
            loggers=loggers,
            show_progress=show_progress,
        )

        # self._step_size = step_size
        # self._step_start = self.step_count
        # self._step_end = self.step_count + n_iter

        with timing.timed_scope(force=timeit) as timer:
            try:
                callbacks.on_run_start(self.step_count, self)
                for step in range(self.step_count, self.step_count + n_iter):
                    self._step_attempt = 0
                    step_log_data = {}

                    while True:
                        callbacks.on_step_start(self.step_count, step_log_data, self)

                        self.reset_step()

                        callbacks.on_compute_update_start(
                            self.step_count, step_log_data, self
                        )
                        self._loss_stats, self._dp = self.compute_loss_and_update()
                        reject_step = callbacks.on_compute_update_end(
                            self.step_count, step_log_data, self
                        )
                        if reject_step:
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

                    callbacks.on_step_end(self.step_count, step_log_data, self)
                    self._step_count += 1

                callbacks.on_run_end(self.step_count, self)
            except StopRun as error:
                callbacks.on_run_end(self.step_count, self)
                print("Stopping early because of : ", error)

            except KeyboardInterrupt as error:
                callbacks.on_run_error(self.step_count, error, self)
                if _graceful_keyboard_interrupt and hasattr(sys, "ps1"):
                    print("Stopped by user.")
                else:
                    raise
            except Exception as error:
                callbacks.on_run_error(self.step_count, error, self)
                raise error

        if timeit:
            self._timer = timer
            if jax.process_count() == 0:
                print(timer)

        return loggers

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
        Updates the parameters of the machine using the optimizer in this driver

        Args:
            dp: the pytree containing the updates to the parameters
        """
        self._optimizer_state, self.state.parameters = apply_gradient(
            self._optimizer.update, self._optimizer_state, dp, self.state.parameters
        )


@partial(jax.jit, static_argnums=0)
def apply_gradient(optimizer_fun, optimizer_state, dp, params):
    import optax

    updates, new_optimizer_state = optimizer_fun(dp, optimizer_state, params)

    new_params = optax.apply_updates(params, updates)
    return new_optimizer_state, new_params
