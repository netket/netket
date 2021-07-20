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

from tqdm import tqdm

import jax
from jax.tree_util import tree_map

from netket.logging import JsonLog
from netket.utils import mpi


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
# - Either _forward_and_backward or individually _forward, _backward, that should
#   compute the loss function and the gradient. If the driver is minimizing or
#   maximising some loss function, this quantity should be assigned to self._stats
#   in order to monitor it.
# - _estimate_stats should return the MC estimate of a single operator
# - reset should reset the driver (usually the sampler).
# - info should return a string with an overview of the driver.
# - The __init__ method shouldbe called with the machine and the optimizer. If this
#   driver is minimising a loss function and you want it's name to show up automatically
#   in the progress bar/ouput files you should pass the optional keyword argument
#   minimized_quantity_name.
class AbstractVariationalDriver(abc.ABC):
    """Abstract base class for NetKet Variational Monte Carlo drivers"""

    def __init__(self, variational_state, optimizer, minimized_quantity_name=""):
        self._mynode = mpi.node_number
        self._mpi_nodes = mpi.n_nodes
        self._loss_stats = None
        self._loss_name = minimized_quantity_name
        self._step_count = 0

        self._variational_state = variational_state
        self.optimizer = optimizer

    def _forward_and_backward(self):  # pragma: no cover
        """
        Performs the forward and backward pass at the same time.
        Concrete drivers should either override this method, or override individually
        _forward and _backward.

        Returns:
            the update for the weights.
        """
        self._forward()
        dp = self._backward()
        return dp

    def _forward(self):
        """
        Performs the forward pass, computing the loss function.
        Concrete should either implement _forward and _backward or the joint method
        _forward_and_backward.
        """
        raise NotImplementedError()  # pragma: no cover

    def _backward(self):
        """
        Performs the backward pass, computing the update for the parameters.
        Concrete should either implement _forward and _backward or the joint method
        _forward_and_backward.
        """
        raise NotImplementedError()  # pragma: no cover

    def _estimate_stats(self, observable):
        """
        Returns the MCMC statistics for the expectation value of an observable.
        Must be implemented by super-classes of AbstractVMC.

        :param observable: A quantum operator (netket observable)
        :return:
        """
        return self.state.expect(observable)

    def reset(self):
        """
        Resets the driver.
        Concrete drivers should also call super().reset() to ensure that the step
        count is set to 0.
        """
        self.state.reset()
        self._step_count = 0
        pass

    @abc.abstractmethod
    def info(self, depth=0):
        """
        Returns an info string used to print information to screen about this driver.
        """
        pass  # pragma: no cover

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
            n_iter: The total number of steps to perform.
            step_size: The number of internal steps the simulation
                is advanced every turn.

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

        steps: (Default=1) number of steps
        """
        for _ in self.iter(steps):
            pass

    def run(
        self,
        n_iter,
        out=None,
        obs=None,
        show_progress=True,
        save_params_every=50,  # for default logger
        write_every=50,  # for default logger
        step_size=1,  # for default logger
        callback=lambda *x: True,
    ):
        """
        Executes the Monte Carlo Variational optimization, updating the weights of the network
        stored in this driver for `n_iter` steps and dumping values of the observables `obs`
        in the output `logger`. If no logger is specified, creates a json file at `out`,
        overwriting files with the same prefix.

        By default uses :ref:`netket.logging.JsonLog`. To know about the output format
        check it's documentation. The logger object is also returned at the end of this function
        so that you can inspect the results without reading the json output.

        Args:
            n_iter: the total number of iterations
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            obs: An iterable containing all observables that should be computed
            save_params_every: Every how many steps the parameters of the network should be
                serialized to disk (ignored if logger is provided)
            write_every: Every how many steps the json data should be flushed to disk (ignored if
                logger is provided)
            step_size: Every how many steps should observables be logged to disk (default=1)
            show_progress: If true displays a progress bar (default=True)
            callback: Callable or list of callable callback functions to stop training given a condition
        """

        if not isinstance(n_iter, numbers.Number):
            raise ValueError(
                "n_iter, the first positional argument to `run`, must be a number!"
            )

        if obs is None:
            obs = {}

        if out is None:
            out = tuple()
            print(
                "No output specified (out=[apath|nk.logging.JsonLogger(...)])."
                "Running the optimization but not saving the output."
            )

        # Log only non-root nodes
        if self._mynode == 0:
            # if out is a path, create an overwriting Json Log for output
            if isinstance(out, str):
                loggers = (JsonLog(out, "w", save_params_every, write_every),)
            else:
                loggers = _to_iterable(out)
        else:
            loggers = tuple()
            show_progress = False

        callbacks = _to_iterable(callback)
        callback_stop = False

        with tqdm(total=n_iter, disable=not show_progress) as pbar:
            old_step = self.step_count
            first_step = True

            for step in self.iter(n_iter, step_size):

                log_data = self.estimate(obs)

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
        return tree_map(self._estimate_stats, observables)

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
