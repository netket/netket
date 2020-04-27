import sys

import numpy as _np
import math

import netket as _nk
from netket._core import deprecated
from .operator import local_values as _local_values
from netket.stats import (
    statistics as _statistics,
    mean as _mean,
    sum_on_nodes as _sum_on_nodes,
)

from netket.vmc_common import info
from netket.abstract_variational_driver import AbstractVariationalDriver


class Vmc(AbstractVariationalDriver):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self, hamiltonian, sampler, optimizer, n_samples, n_discard=None, sr=None
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
        self._sr = sr
        if sr is not None:
            self._sr.is_holomorphic = sampler.machine.is_holomorphic

        self._npar = self._machine.n_par

        self._batch_size = sampler.sample_shape[0]

        self.n_samples = n_samples
        self.n_discard = n_discard

        self._dp = None

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
        self._n_samples_node = int(math.ceil(n_samples_chain / _nk.MPI.size()))

        self._n_samples = int(self._n_samples_node * self._batch_size * _nk.MPI.size())

        self._samples = None

        self._der_logs = None

        self._grads = None

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
            n_discard
            if n_discard != None
            else self._n_samples_node * self._batch_size // 10
        )

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self._sampler.reset()

        # Burnout phase
        self._sampler.generate_samples(self._n_discard)

        # Generate samples and store them
        self._samples = self._sampler.generate_samples(
            self._n_samples_node, samples=self._samples
        )

        # Compute the local energy estimator and average Energy
        eloc, self._loss_stats = self._get_mc_stats(self._ham)

        # Perform update
        if self._sr:
            # When using the SR (Natural gradient) we need to have the full jacobian
            # Computes the jacobian
            _der_logs = self._der_logs
            _der_log = self._machine.der_log
            _samples = self._samples

            _der_logs = _der_log(_samples.reshape((-1, _samples.shape[-1])), _der_logs)

            # Center the local energy
            eloc -= _mean(eloc)

            # Center the log derivatives
            _der_logs -= _mean(_der_logs, axis=0)

            # Compute the gradient
            self._grads = _der_logs.conjugate() * eloc.reshape(-1, 1)

            grad = _mean(self._grads, axis=0)

            self._dp = self._sr.compute_update(_der_logs, grad, self._dp)

            _der_logs = _der_logs.reshape(
                self._n_samples_node, self._batch_size, self._npar
            )
        else:
            # Computing updates using the simple gradient
            # Center the local energy
            eloc -= _mean(eloc)

            self._grads = self._machine.vector_jacobian_prod(
                self._samples, eloc, self._grads
            )

            self._dp = _sum_on_nodes(self._grads) / float(self._n_samples)

        return self._dp

    @property
    def energy(self):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    def _estimate_stats(self, obs):
        return self._get_mc_stats(obs)[1]

    def reset(self):
        self._sampler.reset()
        super().reset()

    def _get_mc_stats(self, op):
        loc = _np.empty((self._samples.shape[0:2]), dtype=_np.complex128)
        for i, sample in enumerate(self._samples):
            _local_values(op, self._machine, sample, out=loc[i])

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
                ("Hamiltonian", self._ham),
                ("Machine", self._machine),
                ("Optimizer", self._optimizer),
                ("SR solver", self._sr),
            ]
        ]
        return "\n  ".join([str(self)] + lines)
