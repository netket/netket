import sys

import numpy as _np

from netket import legacy as _nk
from netket.utils import deprecated
from netket.operator import local_values as _local_values

from .random import randint

from netket.legacy.stats import (
    statistics as _statistics,
    subtract_mean as _subtract_mean,
    mean as _mean,
)

from netket.legacy.vmc_common import info
from netket.legacy.abstract_variational_driver import AbstractVariationalDriver

from numba import jit


class Qsr(AbstractVariationalDriver):
    """
    Quantum State Reconstruction for pure states.
    This implements the algorithm introduced in
    Torlai,et al. Nature Phys 14, 447–450 (2018).
    """

    def __init__(
        self,
        sampler,
        optimizer,
        samples,
        rotations,
        bases,
        n_samples,
        n_samples_data,
        n_discard=None,
        sr=None,
    ):
        """
        Initializes the driver class.

        Args:
            sampler: The Monte Carlo sampler.
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient. This parameter supports three different kinds of inputs,
                which are described in the docs of `make_optimizer_fn`.
            samples: An array of training samples from which the wave function is to be reconstructed.
                Shape is (n_training_samples,hilbert.size).
            rotations: A list of `netket.Operator` defining the unitary rotations defining the basis in which
                the samples are given.
            bases: An array of integers of shape (n_training_samples) containing the index of the corresponding rotation.
                If bases[i]=k, for example, then the sample in samples[i] is measured in the basis defined by rotations[k].
            n_samples (int): Number of sampling sweeps to be
                performed at each step of the optimization when sampling from the model wave-function.
            n_samples_data (int): Number of sampling steps to be
                performed at each step of the optimization when sampling from the given data.
            n_discard (int, optional): Number of sweeps to be discarded at the
                beginning of the sampling, at each step of the optimization.
                Defaults to 10% of the number of samples allocated to each MPI node.
            sr (SR, optional): Determines whether and how stochastic reconfiguration
                is applied to the bare energy gradient before performing applying
                the optimizer. If this parameter is not passed or None, SR is not used.

        """
        super().__init__(sampler.machine, optimizer)

        self._sampler = sampler
        self.sr = sr

        self._rotations = rotations
        self._t_samples = _np.asarray(samples)
        self._bases = _np.asarray(bases)

        self._npar = self._machine.n_par

        self._batch_size = sampler.sample_shape[0]
        self._hilbert = self._machine.hilbert

        self.n_samples = n_samples
        self.n_discard = n_discard

        self.n_samples_data = n_samples_data

        assert self._t_samples.ndim == 2
        for samp in self._t_samples:
            assert samp.shape[0] == self._hilbert.size

        self._n_training_samples = self._t_samples.shape[0]

        assert self._bases.ndim == 1
        assert self._bases.size == self._n_training_samples

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
        self._n_samples = n_samples
        n_samples_chain = int(_np.ceil((n_samples / self._batch_size)))
        self._n_samples_node = int(_np.ceil(n_samples_chain / self._mpi_nodes))

        self._samples = _np.ndarray(
            (self._n_samples_node, self._batch_size, self._hilbert.size)
        )

        self._der_logs = _np.ndarray(
            (self._n_samples_node, self._batch_size, self._npar), dtype=_np.complex128
        )

        self._grads = _np.empty(
            (self._n_samples_node, self._machine.n_par), dtype=_np.complex128
        )

    @property
    def n_samples_data(self):
        return self._n_samples_data

    @n_samples_data.setter
    def n_samples_data(self, n_samples_data):
        if n_samples_data <= 0:
            raise ValueError(
                "Invalid number of samples: n_samples_data={}".format(n_samples)
            )
        self._n_samples_data = n_samples_data
        self._n_samples_data_node = int(_np.ceil(n_samples_data / self._mpi_nodes))

        self._data_grads = _np.empty(
            (self._n_samples_data_node, self._machine.n_par), dtype=_np.complex128
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
        Perform one or several iteration steps of the Qsr calculation. In each step,
        the gradient will be estimated via negative and positive phase and subsequently,
        the variational parameters will be updated according to the configured method.

        Args:
            n_steps (int): Number of steps to perform.
        """

        # Generate samples from the model
        self._sampler.reset()

        # Burnout phase
        for _ in self._sampler.samples(self._n_discard):
            pass

        # Generate samples and store them
        for i, sample in enumerate(self._sampler.samples(self._n_samples_node)):
            self._samples[i] = sample

        # Randomly select a batch of training data
        self._rand_ind = self._get_rand_ind(
            self._n_samples_data_node, self._n_training_samples
        )

        self._data_samples = self._t_samples[self._rand_ind]
        self._data_bases = self._bases[self._rand_ind]

        # Perform update
        if self._sr:
            # When using the SR (Natural gradient) we need to have the full jacobian
            # Computes the jacobian
            for i, sample in enumerate(self._samples):
                self._der_logs[i] = self._machine.der_log(sample, out=self._der_logs[i])

            grad_neg = _mean(self._der_logs.reshape(-1, self._npar), axis=0).conjugate()

            # Positive phase driven by the data
            for x, b_x, grad_x in zip(
                self._data_samples, self._data_bases, self._data_grads
            ):
                self._compute_rotated_grad(x, b_x, grad_x)

            grad_pos = _mean(self._data_grads, axis=0)

            grad = 2.0 * (grad_neg - grad_pos)

            dp = _np.empty(self._npar, dtype=_np.complex128)

            self._sr.compute_update(self._der_logs.reshape(-1, self._npar), grad, dp)
        else:
            # Computing updates using the simple gradient

            # Negative phase driven by the model
            vec_ones = _np.ones(self._batch_size, dtype=_np.complex128) / float(
                self._batch_size
            )
            for x, grad_x in zip(self._samples, self._grads):
                self._machine.vector_jacobian_prod(x, vec_ones, grad_x)

            grad_neg = _mean(self._grads, axis=0)

            # Positive phase driven by the data
            for x, b_x, grad_x in zip(
                self._data_samples, self._data_bases, self._data_grads
            ):
                self._compute_rotated_grad(x, b_x, grad_x)

            grad_pos = _mean(self._data_grads, axis=0)

            dp = 2.0 * (grad_neg - grad_pos)

        return dp

    def _compute_rotated_grad(self, x, basis, out):
        x_primes, mels = self._rotations[basis].get_conn(x)

        log_val_primes = self._machine.log_val(x_primes)

        vec = self._rotated_grad_kernel(log_val_primes, mels)

        self._machine.vector_jacobian_prod(x_primes, vec, out)

    @staticmethod
    @jit
    def _rotated_grad_kernel(log_val_primes, mels):
        vec = _np.empty(mels.size, dtype=_np.complex128)
        max_log_val = log_val_primes.real.max()
        vec = (mels * _np.exp(log_val_primes - max_log_val)).conjugate()
        vec /= vec.sum()
        return vec

    @staticmethod
    @jit
    def _get_rand_ind(n, n_max):
        return _np.asarray(randint(0, n_max, size=(n,)), dtype=_np.intc)

    def _estimate_stats(self, obs):
        return self._get_mc_stats(obs)[1]

    def reset(self):
        self._sampler.reset()
        super().reset()

    def _get_mc_stats(self, op):
        loc = _np.empty(self._samples.shape[0:2], dtype=_np.complex128)
        for i, sample in enumerate(self._samples):
            _local_values(op, self._machine, sample, out=loc[i])
        # notice that loc.T is passed to statistics, since that function assumes
        # that the first index is the batch index.
        return loc, _statistics(loc.T)

    def __repr__(self):
        return "Sqr(step_count={}, n_samples={}, n_discard={})".format(
            self.step_count, self.n_samples, self.n_discard
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Machine", self._machine),
                ("Optimizer", self._optimizer_desc),
                ("SR solver", self._sr),
            ]
        ]
        return "\n  ".join([str(self)] + lines)

    def nll(self, rotations, samples, bases, log_norm=0):
        r"""
        Negative log-likelihood, :math:`\langle log(|Psi_b(x)|^2) \rangle`,
        where the average is over the given samples, and :math:`b` denotes
        the given bases associated to the samples.

        Args:
            rotations: Vector of unitary transformation corresponding to basis rotations.
            samples: Vector of samples.
            bases: Which bases the samples correspond to.
            log_norm: This should be :math:`log \sum_x |\Psi(x)|^2`. Notice that
                      if the probability disitribution is not normalized,
                      (i.e. log_norm :math:`\neq 0`), a user-supplied log_norm must be
                      provided, otherwise there is no guarantuee that the
                      negative log-likelihood computed here is a meaningful
                      quantity.
        """
        nll = 0.0
        for x, basis in zip(samples, bases):
            x_primes, mels = rotations[basis].get_conn(x)

            log_val_primes = self._machine.log_val(x_primes)

            max_log_val = log_val_primes.real.max()
            psi_rotated = (mels * _np.exp(log_val_primes - max_log_val)).sum()

            nll -= _np.log(_np.square(_np.absolute(psi_rotated))) + 2.0 * max_log_val

        nll /= float(len(samples))

        return _np.mean(_np.atleast_1d(nll)) + log_norm
