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

import abc
import numbers
from functools import partial
from typing import Callable, Optional
from collections.abc import Iterable

from tqdm import tqdm

import jax
from jax.tree_util import tree_map

from netket.logging import AbstractLog, JsonLog
from netket.operator import AbstractOperator
from netket.utils import mpi, warn_deprecation


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


# Note: to implement a new Driver (see also _vmc.py for an example)
# If you want to inherit the nice interface of AbstractMCDriver, you should
# subclass it, defining the following methods:
# - _step , that should compute the loss function and the gradient.
#   If the driver is minimizing or maximising some loss function, this quantity
#   should be assigned to self._stats in order to monitor it.
# - reset should reset the driver (usually the sampler).
# - The __init__ method should be called with the machine and the optimizer. If this
#   driver is minimising a loss function and you want it's name to show up automatically
#   in the progress bar/output files you should pass the optional keyword argument
#   minimized_quantity_name.
class AbstractVariationalDriver(abc.ABC):
    """Abstract base class for NetKet Variational Monte Carlo drivers"""

    def __init__(self, variational_state, optimizer, minimized_quantity_name=""):
        self._loss_stats = None
        self._loss_name = minimized_quantity_name
        self._step_count = 0

        self._variational_state = variational_state
        self.optimizer = optimizer

    def _step(self):  # pragma: no cover
        """
        Performs the forward and backward pass at the same time.
        Concrete drivers should either override this method, or override individually
        _forward and _backward.

        Returns:
            the update for the weights.
        """
        # TODO: deprecated december 2023, remove december 2024
        if hasattr(self, "_forward_and_backward"):
            warn_deprecation(
                """
                `AbstractVariationalDriver._forward_and_backward()` has been
                deprecated in favour of the new name `_step()`.

                If you see this error, you probably defined a custom driver. In
                this case, please rename the method from `_forward_and_backward`
                to `_step`.
                """
            )
            return self._forward_and_backward()
        raise NotImplementedError

    def _log_additional_data(self, log_dict: dict, step: int):
        """
        Method to be implemented in sub-classes of AbstractVariationalDriver to
        log additional data at every step.
        This method is called at every iteration when executing with `run`.

        Args:
            log_dict: The dictionary containing all logged data. It must be
                **modified in-place** adding new keys.
            step: the current step number.

        Returns:
            Nothing. The log dictionary should be modified in place.
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
        out: Optional[Iterable[AbstractLog]] = None,
        obs: Optional[dict[str, AbstractOperator]] = None,
        show_progress: bool = True,
        save_params_every: int = 50,  # for default logger
        write_every: int = 50,  # for default logger
        step_size: int = 1,  # for default logger
        callback: Callable[
            [int, dict, "AbstractVariationalDriver"], bool
        ] = lambda *x: True,
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
        """

        if not isinstance(n_iter, numbers.Number):
            raise ValueError(
                "n_iter, the first positional argument to `run`, must be a number!"
            )

        if obs is None:
            obs = {}

        if out is None:
            out = tuple()

        # Log only non-root nodes
        if mpi.node_number == 0 and jax.process_index() == 0:
            # if out is a path, create an overwriting Json Log for output
            if isinstance(out, str):
                out = JsonLog(out, "w", save_params_every, write_every)
            loggers = _to_iterable(out)
        else:
            loggers = tuple()
            show_progress = False

        callbacks = _to_iterable(callback)
        callback_stop = False

        with tqdm(
            total=n_iter,
            disable=not show_progress,
            dynamic_ncols=True,
        ) as pbar:
            old_step = self.step_count
            first_step = True

            for step in self.iter(n_iter, step_size):
                log_data = self.estimate(obs)
                self._log_additional_data(log_data, step)

                # if the cost-function is defined then report it in the progress bar
                if self._loss_stats is not None:
                    pbar.set_postfix_str(self._loss_name + "=" + str(self._loss_stats))
                    log_data[self._loss_name] = self._loss_stats

                # Execute callbacks before loggers because they can append to log_data
                for callback in callbacks:
                    if not callback(step, log_data, self):
                        callback_stop = True

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
        return tree_map(
            lambda O: self.state.expect(O),
            observables,
            is_leaf=lambda x: isinstance(x, AbstractOperator),
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
