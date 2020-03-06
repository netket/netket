import sys

import numpy as _np

import netket as _nk
from netket._core import deprecated
from netket.operator import local_values as _local_values
from netket.operator import _rotated_grad_kernel
from ._C_netket.utils import random_engine, rand_uniform_int

from netket.stats import (
    statistics as _statistics,
    covariance_sv as _covariance_sv,
    subtract_mean as _subtract_mean,
    mean as _mean,
)


def info(obj, depth=None):
    if hasattr(obj, "info"):
        return obj.info(depth)
    else:
        return str(obj)


def make_optimizer_fn(arg, ma):
    """
    Utility function to create the optimizer step function for VMC drivers.

    It currently supports three kinds of inputs:

    1. A NetKet optimizer, i.e., a subclass of `netket.optimizer.Optimizer`.

    2. A 3-tuple (init, update, get) of optimizer functions as used by the JAX
       optimizer submodule (jax.experimental.optimizers).

       The update step p0 -> p1 with bare step dp is computed as
            x0 = init(p0)
            x1 = update(i, dp, x1)
            p1 = get(x1)

    3. A single function update with signature p1 = update(i, dp, p0) returning the
       updated parameter value.
    """
    if isinstance(arg, tuple) and len(arg) == 3:
        init, update, get = arg

        def optimize_fn(i, grad, p):
            x0 = init(p)
            x1 = update(i, grad, x0)
            return get(x1)

        desc = "JAX-like optimizer"
        return optimize_fn, desc

    elif issubclass(type(arg), _nk.optimizer.Optimizer):

        arg.init(ma.n_par, ma.is_holomorphic)

        def optimize_fn(_, grad, p):
            arg.update(grad, p)
            return p

        desc = info(arg)
        return optimize_fn, desc

    elif callable(arg):
        import inspect

        sig = inspect.signature(arg)
        if not len(sig.parameters) == 3:
            raise ValueError(
                "Expected netket.optimizer.Optimizer subclass, JAX optimizer, "
                + " or callable f(i, grad, p); got callable with signature {}".format(
                    sig
                )
            )
        desc = "{}{}".format(arg.__name__, sig)
        return arg, desc
    else:
        raise ValueError(
            "Expected netket.optimizer.Optimizer subclass, JAX optimizer, "
            + " or callable f(i, grad, p); got {}".format(arg)
        )


class Qdmr(object):
    """
    TODO
    """

    def __init__(
        self,
        machine,
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
            machine: The machine representing the density matrix rho(x,y).
            sampler: The Monte Carlo sampler for the diagonal of the densitry matrix rho(x,x).
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
        self._machine = machine
        self._diag_machine = machine.diagonal()
        self._sampler = sampler
        self._sr = sr
        self._stats = None

        self._rotations = rotations
        self._t_samples = _np.asarray(samples)

        self._bases = _np.asarray(bases)

        self._optimizer_step, self._optimizer_desc = make_optimizer_fn(
            optimizer, self._machine
        )

        self._npar = self._machine.n_par

        self._batch_size = sampler.sample_shape[0]
        self._hilbert = self._diag_machine.hilbert

        self.n_samples = n_samples
        self.n_discard = n_discard

        self.n_samples_data = n_samples_data

        self._obs = {}

        self.step_count = 0

        assert self._t_samples.ndim == 2
        for samp in self._t_samples:
            assert samp.shape[0] == self._hilbert.size

        self._n_training_samples = self._t_samples.shape[0]

        assert self._bases.ndim == 1
        assert self._bases.size == self._n_training_samples

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
        self._n_samples_node = int(_np.ceil(n_samples_chain / _nk.MPI.size()))

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
        self._n_samples_data_node = int(_np.ceil(n_samples_data / _nk.MPI.size()))

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

    def advance(self, n_steps=1):
        """
        Perform one or several iteration steps of the Qdmr calculation. In each step,
        the gradient will be estimated via negative and positive phase and subsequently,
        the variational parameters will be updated according to the configured method.

        Args:
            n_steps (int): Number of steps to perform.
        """

        for _ in range(n_steps):

            # Generate samples from the model
            self._sampler.reset()

            # Burnout phase
            for _ in self._sampler.samples(self._n_discard):
                pass

            # Generate samples and store them
            for i, sample in enumerate(self._sampler.samples(self._n_samples_node)):
                self._samples[i] = sample

            # Randomly select a batch of training data
            rand_ind = _np.empty(self._n_samples_data_node, dtype=_np.intc)

            rand_uniform_int(0, (self._n_training_samples - 1), rand_ind)

            self._data_samples = self._t_samples[rand_ind]
            self._data_bases = self._bases[rand_ind]

            # Perform update
            if self._sr:
                # When using the SR (Natural gradient) we need to have the full jacobian
                # Computes the jacobian
                for i, sample in enumerate(self._samples):
                    self._der_logs[i] = self._machine.der_log(sample)

                grad_neg = _mean(
                    self._der_logs.reshape(-1, self._npar), axis=0
                ).conjugate()

                # Positive phase driven by the data
                for x, b_x, grad_x in zip(
                    self._data_samples, self._data_bases, self._data_grads
                ):
                    self._compute_rotated_grad(x, b_x, grad_x)

                grad_pos = _mean(self._data_grads, axis=0)

                grad = 2.0 * (grad_neg - grad_pos)

                dp = _np.empty(self._npar, dtype=_np.complex128)

                self._sr.compute_update(
                    self._der_logs.reshape(-1, self._npar), grad, dp
                )
            else:
                # Computing updates using the simple gradient

                # Negative phase driven by the model
                for i in range(self._samples.shape[0]):
                    self._grads[i] = self._diag_machine.der_log(self._samples[i]).sum(
                        axis=0
                    )

                grad_neg = _mean(self._grads, axis=0) / float(self._batch_size)

                # Positive phase driven by the data
                for i in range(self._data_samples.shape[0]):
                    self._data_grads[i] = self._compute_rotated_grad(
                        self._data_samples[i], self._data_bases[i], self._data_grads[i]
                    )

                grad_pos = _mean(self._data_grads, axis=0)

                dp = grad_neg - grad_pos

            self._machine.parameters = self._optimizer_step(
                self.step_count, dp, self._machine.parameters
            )

            self.step_count += 1

    def _compute_rotated_grad(self, x, basis, out):
        out = _np.zeros(out.shape, dtype=_np.complex128)

        x_primes, mels = self._rotations[basis].get_conn(x)

        log_val_rho = _np.empty(
            (x_primes.shape[0], x_primes.shape[0]), dtype=_np.complex128
        )

        der_log_val_rho = _np.empty(
            (x_primes.shape[0], x_primes.shape[0], self._machine.n_par),
            dtype=_np.complex128,
        )

        for i, x_prime in enumerate(x_primes):
            x_row = _np.tile(x_prime, [x_primes.shape[0], 1])
            log_val_rho[i] = self._machine.log_val(x_row, x_primes)
            der_log_val_rho[i] = self._machine.der_log(x_row, x_primes)

        den = 0.0
        max_log_val = log_val_rho.real.max()

        for i in range(x_primes.shape[0]):
            for j in range(x_primes.shape[0]):
                den_ij = (
                    mels[i]
                    * _np.conjugate(mels[j])
                    * _np.exp(log_val_rho[i, j] - max_log_val)
                )
                den += den_ij
                out += den_ij * der_log_val_rho[i, j]
        out /= den

        return out

    # def _rotated_grad_kernel(self, log_val_primes, mels, vec):
    #     #     max_log_val = log_val_primes.real.max()
    #     #     vec = (mels * _np.exp(log_val_primes - max_log_val)).conjugate()
    #     #     vec /= vec.sum()

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

    def add_observable(self, obs, name):
        """
        Add an observables to the set of observables that will be computed by default
        in get_obervable_stats.
        """
        self._obs[name] = obs

    def get_observable_stats(self, observables=None):
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
        r = {}

        r.update(
            {name: self._get_mc_stats(obs)[1] for name, obs in observables.items()}
        )
        return r

    def reset(self):
        self.step_count = 0
        self._sampler.reset()

    def _get_mc_stats(self, op):
        loc = _np.empty(self._samples.shape[0:2], dtype=_np.complex128)
        for i, sample in enumerate(self._samples):
            _local_values(op, self._machine, sample, out=loc[i])
        return loc, _statistics(loc)

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

    def nll(self, rotations, samples, bases, log_trace=0):
        """
        Negative log-likelihood, :math:`\langle log(\rho_b(x,x)) \rangle`,
        where the average is over the given samples, and :math:`b` denotes
        the given bases associated to the samples.

        Args:
            rotations: Vector of unitary transformation corresponding to basis rotations.
            samples: Vector of samples.
            bases: Which bases the samples correspond to.
            log_trace: This should be :math:`log \sum_x \rho(x,x)`. Notice that
                      if the probability disitribution is not normalized,
                      (i.e. log_trace :math:`\neq 0`), a user-supplied log_trace must be
                      provided, otherwise there is no guarantuee that the
                      negative log-likelihood computed here is a meaningful
                      quantity.
        """
        nll = 0.0
        for x, basis in zip(samples, bases):
            x_primes, mels = rotations[basis].get_conn(x)

            log_val_rho = _np.empty(
                (x_primes.shape[0], x_primes.shape[0]), dtype=_np.complex128
            )

            for i, x_prime in enumerate(x_primes):
                x_row = _np.tile(x_prime, [x_primes.shape[0], 1])
                log_val_rho[i] = self._machine.log_val(x_row, x_primes)

            den = 0.0
            max_log_val = log_val_rho.real.max()

            for i in range(x_primes.shape[0]):
                for j in range(x_primes.shape[0]):
                    den_ij = (
                        mels[i]
                        * _np.conjugate(mels[j])
                        * _np.exp(log_val_rho[i, j] - max_log_val)
                    )
                    den += den_ij

            nll -= _np.log(den) + max_log_val

        nll /= float(len(samples))
        nll = _np.mean(_np.atleast_1d(nll)) + log_trace
        return nll.real

    def test_derivatives(self, epsilon=1e-5):
        """
        Perform one or several iteration steps of the Qdmr calculation. In each step,
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

        self._data_samples = self._t_samples
        self._data_bases = self._bases

        # Negative phase driven by the model
        for i in range(self._samples.shape[0]):
            self._grads[i] = self._diag_machine.der_log(self._samples[i]).sum(axis=0)

        grad_neg = _mean(self._grads, axis=0) / float(self._batch_size)

        # rho_full = self._machine.to_matrix()
        # self._grads = _np.zeros(self._grads.shape, dtype=_np.complex128)
        # grad_neg = _np.zeros(self._machine.n_par, dtype=_np.complex128)
        # for n, state in enumerate(self._hilbert.states()):
        #     grad_neg += self._diag_machine.der_log(state) * rho_full[n, n]

        # grad_neg = self._grads

        # Positive phase driven by the data
        # for x, b_x, grad_x in zip(
        #     self._data_samples, self._data_bases, self._data_grads
        # ):
        #     self._compute_rotated_grad(x, b_x, grad_x)

        for i in range(self._data_samples.shape[0]):
            self._data_grads[i] = self._compute_rotated_grad(
                self._data_samples[i], self._data_bases[i], self._data_grads[i]
            )

        grad_pos = _mean(self._data_grads, axis=0)

        alg_der = grad_neg - grad_pos

        for p in range(alg_der.shape[0]):

            delta = _np.zeros(self._machine.n_par)
            delta[p] = epsilon
            self._machine.parameters += delta

            nll_p = self.nll(
                self._rotations,
                self._data_samples,
                self._data_bases,
                self._machine.log_trace(),
            )
            self._machine.parameters -= 2 * delta
            nll_m = self.nll(
                self._rotations,
                self._data_samples,
                self._data_bases,
                self._machine.log_trace(),
            )
            self._machine.parameters += delta
            num_der = (nll_p - nll_m) / (2.0 * epsilon)

            print(num_der.real, "  ", alg_der[p].real)
