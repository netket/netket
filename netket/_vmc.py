import json
import sys

import numpy as _np
import tqdm
from jax.tree_util import tree_map

import netket as _nk
from netket._core import deprecated
from netket.operator import local_values as _local_values
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


class _JsonLog:
    """
    TODO
    """

    def __init__(
        self, output_prefix, n_iter, obs=None, save_params_every=50, write_every=50
    ):
        self._json_out = {"Output": []}
        self._prefix = output_prefix
        self._write_every = write_every
        self._save_params_every = save_params_every
        self._n_iter = n_iter
        self._obs = obs if obs else {}

    def __call__(self, step, driver):
        item = {"Iteration": step}
        stats = driver.estimate(self._obs)
        stats["Energy"] = driver.energy
        for key, value in stats.items():
            st = value.asdict()
            st["Mean"] = st["Mean"].real
            item[key] = st

        self._json_out["Output"].append(item)

        if step % self._write_every == 0 or step == self._n_iter - 1:
            if _nk.MPI.rank() == 0:
                with open(self._prefix + ".log", "w") as outfile:
                    json.dump(self._json_out, outfile)
        if step % self._save_params_every == 0 or step == self._n_iter - 1:
            if _nk.MPI.rank() == 0:
                driver._machine.save(self._prefix + ".wf")


class Vmc(object):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self, hamiltonian, sampler, optimizer, n_samples, n_discard=None, sr=None
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
        self._ham = hamiltonian
        self._machine = sampler.machine
        self._sampler = sampler
        self._sr = sr
        self._stats = None

        self._optimizer_step, self._optimizer_desc = make_optimizer_fn(
            optimizer, self._machine
        )

        self._npar = self._machine.n_par

        self._batch_size = sampler.sample_shape[0]

        self.n_samples = n_samples
        self.n_discard = n_discard

        self._obs = {}

        self.step_count = 0

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
            (self._n_samples_node, self._batch_size, self._ham.hilbert.size)
        )

        self._der_logs = _np.ndarray(
            (self._n_samples_node, self._batch_size, self._npar), dtype=_np.complex128
        )

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

    def advance(self, n_steps=1):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        for _ in range(n_steps):

            self._sampler.reset()

            # Burnout phase
            for _ in self._sampler.samples(self._n_discard):
                pass

            # Generate samples and store them
            for i, sample in enumerate(self._sampler.samples(self._n_samples_node)):
                self._samples[i] = sample

            # Compute the local energy estimator and average Energy
            eloc, self._stats = self._get_mc_stats(self._ham)

            # Perform update
            if self._sr:
                # When using the SR (Natural gradient) we need to have the full jacobian
                # Computes the jacobian
                for i, sample in enumerate(self._samples):
                    self._der_logs[i] = self._machine.der_log(sample)

                # flatten MC chain dimensions:
                self._der_logs = self._der_logs.reshape(-1, self._npar)

                # Center the local energy
                eloc -= _mean(eloc)

                # Center the log derivatives
                self._der_logs -= _mean(self._der_logs, axis=0)

                # Compute the gradient
                self._grads = _np.conjugate(self._der_logs) * eloc.reshape(-1, 1)

                grad = _mean(self._grads, axis=0)

                dp = _np.empty(self._npar, dtype=_np.complex128)

                self._sr.compute_update(self._der_logs, grad, dp)

                self._der_logs = self._der_logs.reshape(
                    self._n_samples_node, self._batch_size, self._npar
                )
            else:
                # Computing updates using the simple gradient
                # Center the local energy
                eloc -= _mean(eloc)

                for x, eloc_x, grad_x in zip(self._samples, eloc, self._grads):
                    self._machine.vector_jacobian_prod(x, eloc_x, grad_x)

                grad = _mean(self._grads, axis=0) / float(self._batch_size)
                dp = grad

            self._machine.parameters = self._optimizer_step(
                self.step_count, dp, self._machine.parameters
            )

            self.step_count += 1

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
        output = _JsonLog(output_prefix, n_iter, obs, save_params_every, write_every)

        with tqdm.tqdm(
            self.iter(n_iter, step_size), total=n_iter, disable=not show_progress
        ) as itr:
            for step in itr:
                output(step, self)
                itr.set_postfix(Energy=(str(self._stats)))

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

    @property
    def energy(self):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._stats

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

        def estimate(obs):
            return self._get_mc_stats(obs)[1]

        return tree_map(estimate, observables)

    @deprecated()
    def add_observable(self, obs, name):
        """
        Add an observables to the set of observables that will be computed by default
        in get_obervable_stats.
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
        result = self.estimate(observables)
        if include_energy:
            result["Energy"] = self._stats
        return result

    def reset(self):
        self.step_count = 0
        self._sampler.reset()

    def _get_mc_stats(self, op):
        loc = _np.empty(self._samples.shape[0:2], dtype=_np.complex128)
        for i, sample in enumerate(self._samples):
            _local_values(op, self._machine, sample, out=loc[i])

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
