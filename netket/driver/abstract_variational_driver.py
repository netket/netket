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

from typing import Callable, Optional
from collections.abc import Iterable

import abc
import numbers
from functools import partial

from tqdm.auto import tqdm

import jax

from netket.logging import AbstractLog, JsonLog
from netket.operator._abstract_observable import AbstractObservable
from netket.utils import mpi, timing
from netket.utils.types import Optimizer, PyTree
from netket.vqs import VariationalState


def _to_iterable(maybe_iterable):
    """
    _to_iterable(maybe_iterable)

    Ensure the result is iterable. If the input is not iterable, it is wrapped into a tuple.
    """
    if hasattr(maybe_iterable, "__iter__"):
        surely_iterable = maybe_iterable
    else:
        surely_iterable = (maybe_iterable,)

    return surely_iterable


class AbstractVariationalDriver(abc.ABC):
    """Abstract base class for NetKet Variational Monte Carlo drivers

    This class must be inherited from in order to create an optimization driver that
    immediately works with NetKet loggers and callback mechanism.

    .. note::

        How to implement a new driver

        For a concrete example, look at the file `netket/driver/vmc.py`.

        If you want to inherit the nice interface of :class:`netket.driver.AbstractVariationalDriver`,
        you should subclass it, and define the following methods:

        - The :meth:`~netket.driver.AbstractVariationalDriver.__init__` method should be called
          with the machine, optimizer and optionally the name of the loss minimised. If this
          driver is minimising a loss function and you want it's name to show up automatically
          in the progress bar/output files you should pass the optional keyword argument.

        - :meth:`~netket.driver.AbstractVariationalDriver._forward_and_backward`,
          that should compute the loss function and the gradient, returning the latter.
          If the driver is minimizing or maximising some loss function,
          this quantity should be assigned to the field `self._loss_stats`
          in order to monitor it.

        - :meth:`~netket.driver.AbstractVariationalDriver._estimate_stats` should return
          the expectation value over the variational state of a single observable.

        - :meth:`~netket.driver.AbstractVariationalDriver.reset`,
          should reset the driver (usually the sampler). The basic implementation will call
          :meth:`~netket.vqs.VariationalState.reset`, but you are responsible for resetting
          extra fields in the driver itself.

    """

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
        self._mynode = mpi.node_number
        self._is_root = self._mynode == 0 and jax.process_index() == 0
        self._mpi_nodes = mpi.n_nodes
        self._loss_stats = None
        self._loss_name = minimized_quantity_name
        self._step_count = 0
        self._timer = None

        self._variational_state = variational_state
        self.optimizer = optimizer

    def _forward_and_backward(self) -> PyTree:  # pragma: no cover
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

    def _estimate_stats(self, observable):
        """
        Returns the MCMC statistics for the expectation value of an observable.
        Must be implemented by super-classes of AbstractVMC.

        Args:
            observable: A quantum operator (netket observable)

        Returns:
            The expectation value of the observable.
        """
        return self.state.expect(observable)

    def _log_additional_data(self, log_dict: dict, step: int):
        """
        Method to be implemented in sub-classes of AbstractVariationalDriver to
        log additional data at every step.
        This method is called at every iteration when executing with `run`.

        Args:
            log_dict: The dictionary containing all logged data. It must be
                **modified in-place** adding new keys.
            step: the current step number.
        """

    def reset(self):
        """
        Resets the driver.

        Subclasses should make sure to call :code:`super().reset()` to ensure
        that the step count is set to 0.
        """
        self.state.reset()
        self._step_count = 0

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

    def iter(self, n_steps: int, step: int = 1):
        """
        Returns a generator which advances the VMC optimization, yielding
        after every `step_size` steps.

        Args:
            n_steps: The total number of steps to perform (this is
                equivalent to the length of the iterator)
            step: The number of internal steps the simulation
                is advanced between yielding from the iterator

        Yields:
            int: The current step.
        """
        for _ in range(0, n_steps, step):
            for i in range(0, step):
                dp = self._forward_and_backward()
                if i == 0:
                    yield self.step_count

                self._step_count += 1
                self.update_parameters(dp)

    def advance(self, steps: int = 1):
        """
        Performs `steps` optimization steps.

        Args:
            steps: (Default=1) number of steps.

        """
        for _ in self.iter(steps):
            pass

    def run(
        self,
        n_iter: int,
        out: Optional[Iterable[AbstractLog]] = (),
        obs: Optional[dict[str, AbstractObservable]] = None,
        step_size: int = 1,
        show_progress: bool = True,
        save_params_every: int = 50,  # for default logger
        write_every: int = 50,  # for default logger
        callback: Callable[
            [int, dict, "AbstractVariationalDriver"], bool
        ] = lambda *x: True,
        timeit: bool = False,
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
        """

        if not isinstance(n_iter, numbers.Number):
            raise ValueError(
                "n_iter, the first positional argument to `run`, must be a number!"
            )

        if obs is None:
            obs = {}

        # if out is a path, create an overwriting Json Log for output
        if isinstance(out, str):
            out = JsonLog(out, "w", save_params_every, write_every)
        elif out is None:
            out = ()

        # Log only non-root nodes
        if self._is_root:
            loggers = _to_iterable(out)
        else:
            loggers = tuple()
            show_progress = False

        callbacks = _to_iterable(callback)
        callback_stop = False

        with timing.timed_scope(force=timeit) as timer:
            with tqdm(
                total=n_iter,
                disable=not show_progress,
                dynamic_ncols=True,
            ) as pbar:
                old_step = self.step_count
                first_step = True

                for step in self.iter(n_iter, step_size):
                    with timing.timed_scope(name="observables"):
                        log_data = self.estimate(obs)
                        self._log_additional_data(log_data, step)

                    # if the cost-function is defined then report it in the progress bar
                    if self._loss_stats is not None:
                        pbar.set_postfix_str(
                            self._loss_name + "=" + str(self._loss_stats)
                        )
                        log_data[self._loss_name] = self._loss_stats

                    # Execute callbacks before loggers because they can append to log_data
                    for callback in callbacks:
                        if not callback(step, log_data, self):
                            callback_stop = True

                    with timing.timed_scope(name="loggers"):
                        for logger in loggers:
                            logger(self.step_count, log_data, self.state)

                    if len(callbacks) > 0:
                        if mpi.mpi_any(callback_stop):
                            break

                    # Reset the timing of tqdm after the first step, to ignore compilation time
                    if first_step:
                        first_step = False
                        pbar.unpause()

                    # Update the progress bar
                    pbar.update(self.step_count - old_step)
                    old_step = self.step_count

                # Final update so that it shows up filled.
                pbar.update(self.step_count - old_step)

        # flush at the end of the evolution so that final values are saved to
        # file
        for logger in loggers:
            logger.flush(self.state)

        if timeit:
            self._timer = timer
            if self._is_root:
                print(timer)

        return loggers

    def estimate(self, observables):
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
        return jax.tree_util.tree_map(
            self._estimate_stats,
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
