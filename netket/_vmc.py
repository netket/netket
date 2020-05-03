import math
import sys
import warnings

import numpy as _np

import netket as _nk
from netket._core import deprecated
from .operator import local_values as _local_values
from netket.stats import statistics as _statistics, mean as _mean

from netket.vmc_common import info, tree_map
from netket.abstract_variational_driver import AbstractVariationalDriver


# Higher-level VMC functions:
def estimate_expectations(
    ops,
    sampler,
    n_samples,
    n_discard=None,
    compute_gradients=None,
    return_gradients=False,
    return_intermediate=False,
):
    """
    For a pytree of linear operators, computes a statistical estimate of the
    respective expectation values, variances, and optionally gradients of the
    expectation values with respect to the variational parameters.

    The estimate is based on `n_samples` configurations
    obtained from `sampler`.

    Args:
        ops: pytree of linear operators
        sampler: A NetKet sampler
        n_samples: Number of MC samples used to estimate expectation values
        n_discard: Number of MC samples dropped from the start of the
            chain (burn-in). Defaults to `min(n_samples // 10, 100)`.
        return_gradients: Whether to compute the gradients of the observables.
        return_intermediate: Whether to return samples and potentially log-derivatives
            which have been used to compute the VMC statistics.

    Returns:
        A pytree of the same structure as `ops` where the leaves are Stats
        objects containing mean, variance, and MC diagonstics for the corresponding
        operator. If `return_gradients` is True, the leaves are tuples of
        MC stats and the gradients, which are ndarray of shape `(psi.n_par,)`.

        If `return_intermediate` is True, the MC samples and log-derivatives
        are returned as well.
    """
    if compute_gradients is not None:
        warnings.warn(
            "Argument compute_gradients is deprecated, use return_gradients instead.",
            FutureWarning,
        )
        return_gradients = compute_gradients

    psi = sampler.machine

    if not n_discard:
        n_discard = min(n_samples // 10, 100)

    # Burnout phase
    sampler.generate_samples(n_discard)
    # Generate samples
    samples = sampler.generate_samples(n_samples)
    samples_flat = samples.reshape((-1, sampler.sample_shape[-1]))

    if return_gradients:
        der_logs = psi.der_log(samples_flat)

    def estimate(op):
        lvs = _local_values(op, psi, samples_flat)
        stats = _statistics(lvs)

        if return_gradients:
            lvs -= _mean(lvs)
            grad = der_logs.conjugate() * lvs.reshape(-1, 1)
            grad = _mean(grad, axis=0)

            return stats, grad
        else:
            return stats

    mc_result = tree_map(estimate, ops)

    if return_intermediate:
        if return_gradients:
            return mc_result, (samples, der_logs)
        else:
            return mc_result, samples
    else:
        return mc_result


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
        self._n_samples = n_samples
        n_samples_chain = int(math.ceil((n_samples / self._batch_size)))
        self._n_samples_node = int(math.ceil(n_samples_chain / _nk.MPI.size()))

        self._samples = None
        self._der_logs = None
        self._grads = _np.empty(
            (self._n_samples_node, self._machine.n_par), dtype=_np.complex128
        )

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
        Estimates the current gradient via VMC sampling.
        """

        self._sampler.reset()

        # Perform update
        if self._sr:
            loss_data, intermediate = estimate_expectations(
                self._ham,
                self._sampler,
                self._n_samples,
                self._n_discard,
                return_gradients=True,
                return_intermediate=True,
            )
            self._loss_stats, self._grads = loss_data
            self._samples, self._der_logs = intermediate

            self._dp = self._sr.compute_update(self._der_logs, self._grads, self._dp)
        else:
            # Burnout phase
            for _ in self._sampler.samples(self._n_discard):
                pass

            # Generate samples and store them
            self._samples = self._sampler.generate_samples(
                self._n_samples_node, samples=self._samples
            )
            # Compute the local energy estimator and average Energy
            eloc, self._loss_stats = self._get_mc_stats(self._ham)
            # Computing updates using the simple gradient
            # Center the local energy
            eloc -= _mean(eloc)

            for x, eloc_x, grad_x in zip(self._samples, eloc, self._grads):
                self._machine.vector_jacobian_prod(x, eloc_x, grad_x)

            self._dp = _mean(self._grads, axis=0) / float(self._batch_size)

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
