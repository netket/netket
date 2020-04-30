import abc

from netket._core import deprecated, warn_deprecation
import netket as _nk

from netket.logging import JsonLog as _JsonLog

from netket.vmc_common import make_optimizer_fn, tree_map

from tqdm import tqdm

import warnings


def _obs_stat_to_dict(value):
    st = value.asdict()
    st["Mean"] = st["Mean"].real
    return st


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

    def __init__(self, machine, optimizer, minimized_quantity_name=""):
        self._mynode = _nk.MPI.rank()
        self._obs = {}  # to deprecate
        self._loss_stats = None
        self._loss_name = minimized_quantity_name
        self.step_count = 0

        self._machine = machine
        self._optimizer_step, self._optimizer_desc = make_optimizer_fn(
            optimizer, self._machine
        )

    def _forward_and_backward(self):
        """
        Performs the forward and backward pass at the same time.
        Concrete drivers should either override this method, or override individually
        _forward and _backward.

        :return: the update for the weights.
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
        raise NotImplementedError()

    def _backward(self):
        """
        Performs the backward pass, computing the update for the parameters.
        Concrete should either implement _forward and _backward or the joint method
        _forward_and_backward.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _estimate_stats(self, observable):
        """
        Returns the MCMC statistics for the expectation value of an observable.
        Must be implemented by super-classes of AbstractVMC.

        :param observable: A quantum operator (netket observable)
        :return:
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Resets the driver.
        Concrete drivers should also call super().reset() to ensure that the step
        count is set to 0.
        """
        self.step_count = 0
        pass

    @abc.abstractmethod
    def info(self, depth=0):
        """
        Returns an info string used to print information to screen about this driver.
        """
        pass

    @property
    def machine(self):
        """
        Returns the machine that is optimized by this driver.
        """
        return self._machine

    def iter(self, n_steps, step=1):
        """
        Returns a generator which advances the VMC optimization, yielding
        after every `step_size` steps.

        Args:
            :n_iter (int=None): The total number of steps to perform.
            :step_size (int=1): The number of internal steps the simulation
                is advanced every turn.

        Yields:
            int: The current step.
        """
        for _ in range(0, n_steps, step):
            for i in range(0, step):
                dp = self._forward_and_backward()
                if i == 0:
                    yield self.step_count

                self.update_parameters(dp)

    def advance(self, steps=1):
        """
        Performs `steps` optimization steps.

        :param steps: (Default=1) number of steps
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
        output_prefix=None,  # TODO: deprecated
    ):
        """
        Executes the Monte Carlo Variational optimization, updating the weights of the network
        stored in this driver for `n_iter` steps and dumping values of the observables `obs`
        in the output `logger`. If no logger is specified, creates a json file at `output_prefix`,
        overwriting files with the same prefix.

        !! Compatibility v2.1
            Before v2.1 the order of the first two arguments, `n_iter` and `output_prefix` was
            reversed. The reversed ordering will still be supported until v3.0, but is deprecated.

        Args:
            :n_iter: the total number of iterations
            :out: A logger object to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            :obs: An iterable containing all observables that should be computed
            :save_params_every: Every how many steps the parameters of the network should be
            serialized to disk (ignored if logger is provided)
            :write_every: Every how many steps the json data should be flushed to disk (ignored if
            logger is provided)
            :step_size: Every how many steps should observables be logged to disk (default=1)
            :show_progress: If true displays a progress bar (default=True)
            :output_prefix: (Deprecated) The prefix at which json output should be stored (ignored if out
              is provided).
        """

        # TODO Remove this deprecated code in v3.0
        # manage deprecated where argument names are not specified, and
        # prefix is passed as the first positional argument and the number
        # of iterations as a second argument.
        if type(n_iter) is str and type(out) is int:
            n_iter, out = out, n_iter
            warn_deprecation(
                "The positional syntax run(output_prefix, n_iter, **args) is deprecated, use run(n_iter, output_prefix, **args) instead."
            )

        if obs is None:
            # TODO remove the first case after deprecation of self._obs in 3.0
            if len(self._obs) != 0:
                obs = self._obs
            else:
                obs = {}

        # output_prefix is deprecated. out should be used and takes over
        # error out if both are passed
        # TODO: remove in v3.0
        if out is not None and output_prefix is not None:
            raise ValueError(
                "Invalid out and output_prefix arguments. Only one of the two can be passed. Note that output_prefix is deprecated and you should use out."
            )
        elif out is None and output_prefix is not None:
            warn_deprecation(
                "The output_prefix argument is deprecated. Use out instead."
            )
            out = output_prefix

        if out is None:
            print(
                "No output specified (out=[apath|nk.logging.JsonLogger(...)])."
                "Running the optimization but not saving the output."
            )

        # Log only non-root nodes
        if self._mynode == 0:
            # if out is a path, create an overwriting Json Log for output
            if isinstance(out, str):
                logger = _JsonLog(out, "w", save_params_every, write_every)
            else:
                logger = out
        else:
            logger = None

        with tqdm(
            self.iter(n_iter, step_size), total=n_iter, disable=not show_progress
        ) as itr:
            for step in itr:
                # if the cost-function is defined then report it in the progress bar
                if self._loss_stats is not None:
                    itr.set_postfix_str(self._loss_name + "=" + str(self._loss_stats))

                obs_data = self.estimate(obs)

                if self._loss_stats is not None:
                    obs_data[self._loss_name] = self._loss_stats

                log_data = tree_map(_obs_stat_to_dict, obs_data)

                if logger is not None:
                    logger(step, log_data, self.machine)

        # flush at the end of the evolution so that final values are saved to
        # file
        if logger is not None:
            logger.flush(self.machine)

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
            :param dp: the gradient
        """
        self._machine.parameters = self._optimizer_step(
            self.step_count, dp, self._machine.parameters
        )
        self.step_count += 1

    @deprecated()
    def add_observable(self, obs, name):
        """
        Add an observables to the set of observables that will be computed by default
        in get_obervable_stats.

        This function is deprecated in favour of `estimate`.

        Args:
            obs: the operator encoding the observable
            name: a string, representing the name of the observable
        """
        self._obs[name] = obs

    @deprecated()
    def get_observable_stats(self, observables=None, include_energy=True):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.

        Args:
            observables: A dictionary of the form {name: observable} or a list
                of tuples (name, observable) for which statistics should be computed.
                If observables is None or not passed, results for those observables
                added to the driver by add_observables are computed.
            include_energy: Whether to include the energy estimate (which is already
                computed as part of the VMC step) in the result.

        Returns:
            A dictionary of the form {name: stats} mapping the observable names in
            the input to corresponding Stats objects.

            If `include_energy` is true, then the result will further contain the
            energy statistics with key "Energy".
        """
        if not observables:
            observables = self._obs

        if self._loss_name is None:
            include_energy = False

        result = self.estimate(observables)

        if include_energy:
            result[self._loss_name] = self._loss_stats

        return result
