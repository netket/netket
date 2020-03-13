import abc

from netket._core import deprecated
import netket as _nk

from netket.vmc_common import tree_map
from netket.vmc_json import _JsonLog


from tqdm import tqdm

class AbstractMCDriver(abc.ABC):
    """Abstract base class for NetKet Variational Monte Carlo runners"""

    def __init__(self, minimized_quantity_name=''):
        self._mynode = _nk.MPI.rank()
        self._obs    = {} # to deprecate
        self._stats  = None
        self._stats_name = minimized_quantity_name
        self.step_count = 0

    @abc.abstractmethod
    def advance(self, step_size):
        pass

    @abc.abstractmethod
    def reset(self):
        self.step_count = 0
        pass

    def run(
            self,
            n_iter,
            output_prefix,
            obs=None,
            save_params_every=50,
            write_every=50,
            step_size=1,
            show_progress=True,
    ):
        """
        TODO
        """
        if obs is None:
            obs = self._obs

        output = _JsonLog(output_prefix, n_iter, obs, save_params_every, write_every)

        with tqdm(
                self.iter(n_iter, step_size), total=n_iter, disable=not show_progress
        ) as itr:
            for step in itr:
                output(step, self)
                if self._stats is not None:
                    itr.set_postfix_str(self._stats_name + ' = ' + str(self._stats))

    def iter(self, n_steps, step=1):
        """
        Returns a generator which advances the VMC optimization, yielding
        after every `step_size` steps.

        Args:
            n_iter (int=None): The total number of steps to perform.
            step_size (int=1): The number of internal steps the simulation
                is advanced every turn.

        Yields:
            int: The current step.
        """
        for _ in range(0, n_steps, step):
            self.advance(step)
            yield self.step_count


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

    @deprecated()
    def add_observable(self, obs, name):
        """
        Add an observables to the set of observables that will be computed by default
        in get_obervable_stats.
        """
        self._obs[name] = obs


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
        result = self.estimate(observables)
        if include_energy:
            result["Energy"] = self._stats
        return result
