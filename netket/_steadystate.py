import sys
import itertools

import numpy as _np

import netket as _nk
from netket._core import deprecated
from ._C_netket import MPI as _MPI
from netket.operator import local_values as _local_values
from netket.operator import der_local_values as _der_local_values
from netket.stats import (
    statistics as _statistics,
    covariance_sv as _covariance_sv,
    mean as _mean,
    subtract_mean as _subtract_mean,
)

import json


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


class SteadyState(object):
    """
    Energy minimization using Variational Monte Carlo (VMC).
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
            hamiltonian: The Hamiltonian of the system.
            sampler: The Monte Carlo sampler.
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

        Example:
            Optimizing a 1D wavefunction with Variational Monte Carlo.

            ```python
            >>> import netket as nk
            >>> SEED = 3141592
            >>> g = nk.graph.Hypercube(length=8, n_dim=1)
            >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            >>> ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
            >>> ma.init_random_parameters(seed=SEED, sigma=0.01)
            >>> ha = nk.operator.Ising(hi, h=1.0)
            >>> sa = nk.sampler.MetropolisLocal(machine=ma)
            >>> sa.seed(SEED)
            >>> op = nk.optimizer.Sgd(learning_rate=0.1)
            >>> vmc = nk.Vmc(ha, sa, op, 200)
            ```
        """
        self._lind = lindblad
        self._machine = sampler.machine
        self._machine_obs = sampler_obs.machine
        self._sampler = sampler
        self._sampler_obs = sampler_obs
        self._sr = sr
        self._stats = None
        self._mynode = _MPI.rank()

        self._optimizer_step, self._optimizer_desc = make_optimizer_fn(
            optimizer, self._machine
        )

        self._npar = self._machine.n_par

        self._batch_size = sampler.sample_shape[0]
        self._batch_size_obs = sampler.sample_shape[0]

        self.n_samples = n_samples
        self.n_discard = n_discard
        self.n_samples_obs = n_samples_obs
        self.n_discard_obs = n_discard_obs

        self._obs = {}

        self.step_count = 0
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
            (self._n_samples_node, self._batch_size, self._npar), dtype=_np.complex128
        )

        self._der_loc_vals = _np.ndarray(
            (self._n_samples_node, self._batch_size, self._npar), dtype=_np.complex128
        )

        # Set the machine_func of the sampler over the diagonal of the density matrix
        # to be |\rho(x,x)|
        sampler_obs.machine_func = lambda x, out: _np.abs(x, out)

    @property
    def n_samples(self):
        return self._n_samples

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
                "Invalid number of discarded samples: n_discard={}".format(n_discard)
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
                "Invalid number of discarded samples: n_discard={}".format(n_discard)
            )
        self._n_discard_obs = (
            n_discard
            if n_discard != None
            else self._n_samples_obs_node * self._batch_size_obs // 10
        )

    def advance(self, n_steps=1):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        for _ in range(n_steps):

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
            lloc, self._stats = self._get_mc_superop_stats(self._lind)

            # Compute the (MPI-aware-)average of the derivatives
            der_logs_ave = _mean(self._der_logs, axis=0)

            # Center the log derivatives
            self._der_logs -= der_logs_ave

            # Compute the gradient
            grad = _covariance_sv(lloc, self._der_loc_vals, center_s=False)
            grad -= self._stats.mean * der_logs_ave.conj()

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

            self._machine.parameters = self._optimizer_step(
                self.step_count, dp, self._machine.parameters
            )

            self.step_count += 1

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

    def iter(self, n_iter=None, step_size=1):
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
        for i in itertools.count(step=step_size):
            if n_iter and i >= n_iter:
                return
            self.advance(step_size)
            yield i

    def add_observable(self, obs, name):
        """
        Add an observables to the set of observables that will be computed by default
        in get_obervable_stats.
        """
        self._obs[name] = obs

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
        r = {"LdagL": self._stats} if include_energy else {}

        r.update(
            {name: self._get_mc_obs_stats(obs)[1] for name, obs in observables.items()}
        )
        return r

    def reset(self):
        self.step_count = 0
        self._sampler.reset()

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
        return "Vmc(step_count={}, n_samples={}, n_discard={})".format(
            self.step_count, self.n_samples, self.n_discard
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Hamiltonian", self._ham),
                ("Machine", self._machine),
                ("Optimizer", self._optimizer_desc),
                ("SR solver", self._sr),
            ]
        ]
        return "\n  ".join([str(self)] + lines)

    def _add_to_json_log(self, step_count):
        stats = self.get_observable_stats()
        self._json_out["Output"].append({})
        self._json_out["Output"][-1] = {}
        json_iter = self._json_out["Output"][-1]
        json_iter["Iteration"] = step_count
        for key, value in stats.items():
            st = value.asdict()
            st["Mean"] = st["Mean"].real
            json_iter[key] = st

    def _init_json_log(self):

        self._json_out = {}
        self._json_out["Output"] = []

    def run(
        self, output_prefix, n_iter, step_size=1, save_params_every=50, write_every=50
    ):
        self._init_json_log()

        for k in range(n_iter):
            self.advance(step_size)

            self._add_to_json_log(k)
            if k % write_every == 0 or k == n_iter - 1:
                if self._mynode == 0:
                    with open(output_prefix + ".log", "w") as outfile:
                        json.dump(self._json_out, outfile)
