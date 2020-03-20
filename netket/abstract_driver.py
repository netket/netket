import abc

from netket._core import deprecated
import netket as _nk

from netket.logging import JsonLog as _JsonLog

from netket.vmc_common import info, make_optimizer_fn, tree_map

from tqdm import tqdm

import warnings


def obs_stat_to_dict(value):
    st = value.asdict()
    st["Mean"] = st["Mean"].real
    return st


class AbstractMCDriver(abc.ABC):
    """Abstract base class for NetKet Variational Monte Carlo drivers"""

    def __init__(self, machine, optimizer, minimized_quantity_name=""):
        self._mynode = _nk.MPI.rank()
        self._obs = {}  # to deprecate
        self._stats = None
        self._stats_name = minimized_quantity_name
        self.step_count = 0

        self._machine = machine
        self._optimizer_step, self._optimizer_desc = make_optimizer_fn(
            optimizer, self._machine
        )

    @abc.abstractmethod
    def gradient(self):
        pass

    @abc.abstractmethod
    def reset(self):
        self.step_count = 0
        pass

    @property
    def machine(self):
        return self._machine

    def run(
        self,
        output_prefix,
        n_iter,
        obs=None,
        save_params_every=50,
        write_every=50,
        step_size=1,
        show_progress=True,
    ):
        """
        Executes the Monte Carlo Variational optimization, updating the weights of the network
        stored in this driver for `n_iter` steps and dumping values of the observables `obs`
        in the output. The output is a json file at `output_prefix`, overwriting files with
        the same prefix.

        Args:
            :output_prefix: The prefix at which json output should be stored (ignored if logger
              is provided).
            :n_iter: the total number of iterations
            :obs: An iterable containing all observables that should be computed
            :save_params_every: Every how many steps the parameters of the network should be
            serialized to disk (ignored if logger is provided)
            :write_every: Every how many steps the json data should be flushed to disk (ignored if
            logger is provided)
            :step_size: Every how many steps should observables be logged to disk (default=1)
            :show_progress: If true displays a progress bar (default=True)
        """
        if obs is None:
            # TODO remove the first case after deprecation of self._obs in 3.0
            if len(self._obs) is not 0:
                obs = self._obs
            else:
                obs = {}

        logger = _JsonLog(output_prefix, save_params_every, write_every)

        # Don't log on non-root nodes
        if self._mynode is not 0:
            logger = None

        with tqdm(
            self.iter(n_iter, step_size), total=n_iter, disable=not show_progress
        ) as itr:
            for step in itr:
                # if the cost-function is defined then report it in the progress bar
                if self._stats is not None:
                    itr.set_postfix_str(self._stats_name + "=" + str(self._stats))

                obs_data = self.estimate(obs)

                if self._stats is not None:
                    obs_data[self._stats_name] = self._stats

                log_data = tree_map(obs_stat_to_dict, obs_data)

                if logger is not None:
                    logger(step, log_data, self.machine)

        # flush at the end of the evolution so that final values are saved to
        # file
        if logger is not None:
            logger.flush(self.machine)

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
                dp = self.gradient()
                if i is 0:
                    yield self.step_count

                self.update_parameters(dp)

    def advance(self, steps=1):
        """
        Performs `steps` optimization steps.

        :param steps: (Default=1) number of steps
        """
        for _ in self.iter(steps):
            pass

    @abc.abstractmethod
    def info(self, depth=0):
        pass

    @abc.abstractmethod
    def estimate_stats(self, observable):
        """
        Returns the MCMC statistics for the expectation value of an observable.
        Must be implemented by super-classes of AbstractVMC.

        :param observable: A quantum operator (netket observable)
        :return:
        """
        pass

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
        return tree_map(self.estimate_stats, observables)

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

        if self._stats_name is None:
            include_energy = False

        result = self.estimate(observables)

        if include_energy:
            result[self._stats_name] = self._stats

        return result
