import math

from netket import legacy as _nk

from netket.operator import local_values as _local_values
from netket.legacy.stats import (
    statistics as _statistics,
    mean as _mean,
    sum_inplace as _sum_inplace,
)

from netket.legacy.vmc_common import info, tree_map
from netket.legacy.abstract_variational_driver import AbstractVariationalDriver


class Vmc(AbstractVariationalDriver):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self,
        hamiltonian,
        sampler,
        optimizer,
        n_samples,
        n_discard=None,
        sr=None,
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian (AbstractOperator): The Hamiltonian of the system.
            sampler: The Monte Carlo sampler.
            optimizer (AbstractOptimizer): Determines how optimization steps are performed given the
                bare energy gradient.
            n_samples (int): Number of Markov Chain Monte Carlo sweeps to be
                performed at each step of the optimization.
            n_discard (int, optional): Number of sweeps to be discarded at the
                beginning of the sampling, at each step of the optimization.
                Defaults to 10% of the number of samples allocated to each MPI node.
            sr (SR, optional): Determines whether and how stochastic reconfiguration
                is applied to the bare energy gradient before performing applying
                the optimizer. If this parameter is not passed or None, SR is not used.

        Example:
            Optimizing a 1D wavefunction with Variational Monte Carlo.

            >>> import netket as nk
            >>> SEED = 3141592
            >>> g = nk.graph.Hypercube(length=8, n_dim=1)
            >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            >>> ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
            >>> ma.init_random_parameters(seed=SEED, sigma=0.01)
            >>> ha = nk.operator.Ising(hi, h=1.0)
            >>> sa = nk.sampler.MetropolisLocal(machine=ma)
            >>> op = nk.optimizer.Sgd(learning_rate=0.1)
            >>> vmc = nk.Vmc(ha, sa, op, 200)

        """
        super(Vmc, self).__init__(
            sampler.machine, optimizer, minimized_quantity_name="Energy"
        )

        self._ham = hamiltonian
        self._sampler = sampler
        self.sr = sr

        self._npar = self._machine.n_par

        self._batch_size = sampler.sample_shape[0]

        # Check how many parallel nodes we are running on
        self.n_nodes = _nk.utils.n_nodes

        self.n_samples = n_samples
        self.n_discard = n_discard

        self._dp = None

    @property
    def sr(self):
        return self._sr

    @sr.setter
    def sr(self, sr):
        self._sr = sr
        if self._sr is not None:
            self._sr.setup(self.machine)

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n_samples):
        if n_samples <= 0:
            raise ValueError(
                "Invalid number of samples: n_samples={}".format(n_samples)
            )

        n_samples_chain = int(math.ceil((n_samples / self._batch_size)))
        self._n_samples_node = int(math.ceil(n_samples_chain / self.n_nodes))

        self._n_samples = int(self._n_samples_node * self._batch_size * self.n_nodes)

        self._samples = None

        self._grads = None
        self._jac = None

    @property
    def n_discard(self):
        return self._n_discard

    @n_discard.setter
    def n_discard(self, n_discard):
        if n_discard is not None and n_discard < 0:
            raise ValueError(
                "Invalid number of discarded samples: n_discard={}".format(n_discard)
            )
        self._n_discard = (
            int(n_discard)
            if n_discard != None
            else self._n_samples_node * self._batch_size // 10
        )

    def _forward_and_backward(self, sample=True):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        if sample:
            self._sampler.reset()

            # Burnout phase
            self._sampler.generate_samples(self._n_discard)

            # Generate samples and store them
            self._samples = self._sampler.generate_samples(
                self._n_samples_node, samples=self._samples
            )

        # Compute the local energy estimator and average Energy
        eloc, self._loss_stats = self._get_mc_stats(self._ham)

        # Center the local energy
        eloc -= _mean(eloc)

        samples_r = self._samples.reshape((-1, self._samples.shape[-1]))
        eloc_r = eloc.reshape(-1, 1)

        # Perform update
        if self._sr:
            if self._sr.onthefly:

                self._grads = self._machine.vector_jacobian_prod(
                    samples_r, eloc_r / self._n_samples, self._grads
                )

                self._grads = tree_map(_sum_inplace, self._grads)

                self._dp = self._sr.compute_update_onthefly(
                    samples_r, self._grads, self._dp
                )

            else:
                # When using the SR (Natural gradient) we need to have the full jacobian
                self._grads, self._jac = self._machine.vector_jacobian_prod(
                    samples_r,
                    eloc_r / self._n_samples,
                    self._grads,
                    return_jacobian=True,
                )

                self._grads = tree_map(_sum_inplace, self._grads)

                self._dp = self._sr.compute_update(self._jac, self._grads, self._dp)

        else:
            # Computing updates using the simple gradient
            self._grads = self._machine.vector_jacobian_prod(
                samples_r, eloc_r / self._n_samples, self._grads
            )

            self._grads = tree_map(_sum_inplace, self._grads)

            #  if Real pars but complex gradient, take only real part
            # not necessary for SR because sr already does it.
            if not self._machine.has_complex_parameters:
                self._dp = tree_map(lambda x: x.real, self._grads)
            else:
                self._dp = self._grads

        return self._dp

    @property
    def energy(self):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    def _estimate_stats(self, obs):
        if self._samples is None:
            raise RuntimeError(
                "Vmc driver needs to perform a step before .estimate() can be "
                "called. To get VMC estimates outside of optimization, use "
                "netket.variational.estimate_expectations instead."
            )
        return self._get_mc_stats(obs)[1]

    def reset(self):
        self._sampler.reset()
        super().reset()

    def _get_mc_stats(self, op):

        samples_r = self._samples.reshape((-1, self._samples.shape[-1]))

        loc = _local_values(op, self._machine, samples_r).reshape(
            self._samples.shape[0:2]
        )

        # notice that loc.T is passed to statistics, since that function assumes
        # that the first index is the batch index.
        return loc, _statistics(loc.T)

    def __repr__(self):
        return "Vmc(step_count={}, n_samples={}, n_discard={})".format(
            self.step_count, self.n_samples, self.n_discard
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Hamiltonian ", self._ham),
                ("Machine     ", self._machine),
                ("Optimizer   ", self._optimizer),
                ("SR solver   ", self._sr),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)
