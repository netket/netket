import numpy as _np

import netket as _nk
from ._C_netket import MPI as _MPI
from .operator import local_values as _local_values
from .operator import der_local_values as _der_local_values
from netket.stats import (
    statistics as _statistics,
    covariance_sv as _covariance_sv,
    mean as _mean,
    subtract_mean as _subtract_mean,
)

from netket.vmc_common import info
from netket.abstract_variational_driver import AbstractVariationalDriver


class SteadyState(AbstractVariationalDriver):
    """
    Variational steady-state search by minimization of the L2,2 norm of L\rho using
    Variational Monte Carlo (VMC).
    """

    def __init__(
        self,
        lindblad,
        sampler,
        optimizer,
        n_samples,
        n_discard=None,
        sr=None,
        sampler_obs=None,
        n_samples_obs=None,
        n_discard_obs=None,
    ):
        """
        Initializes the driver class.

        Args:
            liouvillian: The liouvillian of the system.
            sampler: The Monte Carlo sampler for the density matrix.
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient. This parameter supports three different kinds of inputs,
                which are described in the docs of `make_optimizer_fn`.
            n_samples: Number of Markov Chain Monte Carlo sweeps to be
                performed at each step of the optimization.
            n_discard (int, optional): Number of sweeps to be discarded at the
                beginning of the sampling, at each step of the optimization.
                Defaults to 10% of the number of samples allocated to each MPI node.
            sr (SR, optional): Determines whether and how stochastic reconfiguration
                is applied to the bare energy gradient before performing applying
                the optimizer. If this parameter is not passed or None, SR is not used.
            sampler_obs: The Monte Carlo sampler for the diagonal of the density matrix, used
                to compute observables (default: same as sampler).
            n_samples_obs: n_samples for the observables (default: n_samples)
            n_discard_obs: n_discard for the observables (default: n_discard)

        """
        super(SteadyState, self).__init__(
            sampler.machine, optimizer, minimized_quantity_name="LdagL"
        )  #'\u3008L\u2020L\u3009')

        self._lind = lindblad
        self._machine_obs = sampler_obs.machine
        self._sampler = sampler
        self._sampler_obs = sampler_obs
        self._sr = sr
        if sr is not None:
            self._sr.is_holomorphic = sampler.machine.is_holomorphic

        self._npar = self._machine.n_par

        self._batch_size = sampler.sample_shape[0]
        self._batch_size_obs = sampler.sample_shape[0]

        self.n_samples = n_samples
        self.n_discard = n_discard
        self.n_samples_obs = n_samples_obs
        self.n_discard_obs = n_discard_obs

        self._obs_samples_valid = False
        self._samples = _np.ndarray(
            (self._n_samples_node, self._batch_size, lindblad.hilbert.size)
        )
        self._samples_obs = _np.ndarray(
            (
                self._n_samples_obs_node,
                self._batch_size_obs,
                lindblad.hilbert.size_physical,
            )
        )

        self._der_logs = _np.ndarray(
            (self._n_samples_node, self._batch_size,
             self._npar), dtype=_np.complex128
        )

        self._der_loc_vals = _np.ndarray(
            (self._n_samples_node, self._batch_size,
             self._npar), dtype=_np.complex128
        )

        # Set the machine_pow of the sampler over the diagonal of the density matrix
        # to be |\rho(x,x)|
        sampler_obs.machine_pow = 1.0

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def ldagl(self):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    @property
    def n_samples_obs(self):
        return self._n_samples_obs

    @n_samples.setter
    def n_samples(self, n_samples):
        if n_samples <= 0:
            raise ValueError(
                "Invalid number of samples: n_samples={}".format(n_samples)
            )
        self._n_samples = n_samples
        n_samples_chain = int(_np.ceil((n_samples / self._batch_size)))
        self._n_samples_node = int(_np.ceil(n_samples_chain / _MPI.size()))

    @n_samples_obs.setter
    def n_samples_obs(self, n_samples):
        if self._sampler_obs is None:
            return

        if n_samples <= 0:
            raise ValueError(
                "Invalid number of samples: n_samples={}".format(n_samples)
            )
        self._n_samples_obs = n_samples
        n_samples_chain = int(_np.ceil((n_samples / self._batch_size_obs)))
        self._n_samples_obs_node = int(_np.ceil(n_samples_chain / _MPI.size()))

    @property
    def n_discard(self):
        return self._n_discard

    @property
    def n_discard_obs(self):
        return self._n_discard_obs

    @n_discard.setter
    def n_discard(self, n_discard):
        if n_discard is not None and n_discard < 0:
            raise ValueError(
                "Invalid number of discarded samples: n_discard={}".format(
                    n_discard)
            )
        self._n_discard = (
            n_discard
            if n_discard != None
            else self._n_samples_node * self._batch_size // 10
        )

    @n_discard_obs.setter
    def n_discard_obs(self, n_discard):
        if self._sampler_obs is None:
            return

        if n_discard is not None and n_discard < 0:
            raise ValueError(
                "Invalid number of discarded samples: n_discard={}".format(
                    n_discard)
            )
        self._n_discard_obs = (
            n_discard
            if n_discard is not None
            else self._n_samples_obs_node * self._batch_size_obs // 10
        )

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self._sampler.reset()
        self._obs_samples_valid = False

        # Burnout phase
        for _ in self._sampler.samples(self._n_discard):
            pass

        # Generate samples
        for i, sample in enumerate(self._sampler.samples(self._n_samples_node)):

            # Store the current sample
            self._samples[i] = sample

            # Compute Log derivatives
            self._der_logs[i] = self._machine.der_log(sample)

            self._der_loc_vals[i] = _der_local_values(
                self._lind, self._machine, sample, center_derivative=False
            )

        # flatten MC chain dimensions:
        self._der_logs = self._der_logs.reshape(-1, self._npar)

        # Estimate energy
        lloc, self._loss_stats = self._get_mc_superop_stats(self._lind)

        # Compute the (MPI-aware-)average of the derivatives
        der_logs_ave = _mean(self._der_logs, axis=0)

        # Center the log derivatives
        self._der_logs -= der_logs_ave

        # Compute the gradient
        grad = _covariance_sv(lloc, self._der_loc_vals, center_s=False)
        grad -= self._loss_stats.mean * der_logs_ave.conj()

        # Perform update
        if self._sr:
            dp = _np.empty(self._npar, dtype=_np.complex128)
            # flatten MC chain dimensions:
            derlogs = self._der_logs.reshape(-1, self._der_logs.shape[-1])
            self._sr.compute_update(derlogs, grad, dp)
            derlogs = self._der_logs.reshape(
                self._n_samples_node, self._batch_size, self._npar
            )
        else:
            dp = grad

        self._der_logs = self._der_logs.reshape(
            self._n_samples_node, self._batch_size, self._npar
        )

        return dp

    def sweep_diagonal(self):
        """
        Sweeps the diagonal of the density matrix with the observable sampler.
        """
        self._sampler_obs.reset()

        # Burnout phase
        for _ in self._sampler_obs.samples(self._n_discard_obs):
            pass

        # Generate samples
        for i, sample in enumerate(self._sampler_obs.samples(self._n_samples_obs_node)):

            # Store the current sample
            self._samples_obs[i] = sample

        self._obs_samples_valid = True

    def _estimate_stats(self, obs):
        return self._get_mc_obs_stats(obs)[1]

    def reset(self):
        self._sampler.reset()
        super().reset()

    def _get_mc_superop_stats(self, op):
        loc = _np.empty(self._samples.shape[0:2], dtype=_np.complex128)
        for i, sample in enumerate(self._samples):
            _local_values(op, self._machine, sample, out=loc[i])

        return loc, _statistics(_np.square(_np.abs(loc), dtype="complex128"))

    def _get_mc_obs_stats(self, op):
        if not self._obs_samples_valid:
            self.sweep_diagonal()

        loc = _local_values(op, self._machine, self._samples_obs)
        return loc, _statistics(loc)

    def __repr__(self):
        return "SteadyState(step_count={}, n_samples={}, n_discard={})".format(
            self.step_count, self.n_samples, self.n_discard
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Liouvillian", self._lind),
                ("Machine", self._machine),
                ("Optimizer", self._optimizer_desc),
                ("SR solver", self._sr),
            ]
        ]
        return "\n  ".join([str(self)] + lines)
