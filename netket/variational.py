import itertools
from ._vmc import Vmc as _Vmc
from ._C_netket import MPI as _MPI
from ._C_netket.optimizer import SR as _SR
import json
import warnings
import numpy as _np
from tqdm import trange
from jax.tree_util import tree_map


class Vmc(object):
    def __init__(
        self,
        hamiltonian,
        sampler,
        optimizer,
        n_samples,
        discarded_samples=None,
        discarded_samples_on_init=0,
        target="energy",
        method="Sr",
        diag_shift=0.01,
        use_iterative=False,
        use_cholesky=None,
        sr_lsq_solver="LLT",
    ):

        self._mynode = _MPI.rank()
        self.machine = sampler.machine

        if method == "Gd":
            self.sr = None
            self._vmc = _Vmc(
                hamiltonian=hamiltonian,
                sampler=sampler,
                optimizer=optimizer,
                n_samples=n_samples,
                n_discard=discarded_samples,
                sr=None,
            )
        elif method == "Sr":
            self.sr = _SR(
                lsq_solver=sr_lsq_solver,
                diag_shift=diag_shift,
                use_iterative=use_iterative,
                is_holomorphic=sampler.machine.is_holomorphic,
            )
            self._vmc = _Vmc(
                hamiltonian=hamiltonian,
                sampler=sampler,
                optimizer=optimizer,
                n_samples=n_samples,
                n_discard=discarded_samples,
                sr=self.sr,
            )
        else:
            raise ValueError("Allowed method options are Gd and Sr")

        if use_cholesky and sr_lsq_solver != "LLT":
            raise ValueError(
                "Inconsistent options specified: `use_cholesky && sr_lsq_solver != 'LLT'`."
            )

        if discarded_samples_on_init != 0:
            warnings.warn(
                "discarded_samples_on_init does not have any effect and should not be used",
                DeprecationWarning,
            )

        self.advance = self._vmc.advance
        self.add_observable = self._vmc.add_observable
        self.get_observable_stats = self._vmc.get_observable_stats
        self.reset = self._vmc.reset

        warnings.warn(
            "netket.variational.Vmc will be deprecated in version 3, use netket.Vmc instead",
            PendingDeprecationWarning,
        )

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
        self,
        output_prefix,
        n_iter,
        step_size=1,
        save_params_every=50,
        write_every=50,
        show_progress=True,
    ):
        self._init_json_log()

        with trange(n_iter, disable=not show_progress) as itr:
            for k in itr:
                self.advance(step_size)

                self._add_to_json_log(k)
                if k % write_every == 0 or k == n_iter - 1:
                    if self._mynode == 0:
                        with open(output_prefix + ".log", "w") as outfile:
                            json.dump(self._json_out, outfile)
                if k % save_params_every == 0 or k == n_iter - 1:
                    if self._mynode == 0:
                        self.machine.save(output_prefix + ".wf")

                # Update energy displayed in the progress bar
                itr.set_postfix(Energy=(str(self._vmc._stats)))

    def iter(self, n_iter=None, step_size=1):
        """

        Returns a generator which advances the VMC optimization, yielding
        after every step_size steps up to n_iter.

        Args:
            n_iter (int=None): The number of steps or None, for no limit.
            step_size (int=1): The number of steps the simulation is advanced.

        Yields:
            int: The current step.
        """
        self.reset()
        for i in itertools.count(step=step_size):
            if n_iter and i >= n_iter:
                return
            self.advance(step_size)
            yield i


# Higher-level VMC functions:


def estimate_expectations(
    ops, sampler, n_samples, n_discard=None, compute_gradients=False
):
    """
    For a sequence of linear operators, computes a statistical estimate of the
    respective expectation values, variances, and optionally gradients of the
    expectation values with respect to the variational parameters.

    The estimate is based on `n_samples` configurations
    obtained from `sampler`.

    Args:
        ops: pytree of linear operators
        sampler: A NetKet sampler
        n_samples: Number of MC samples used to estimate expectation values
        n_discard: Number of MC samples dropped from the start of the
            chain (burn-in). Defaults to `n_samples //10`.
        compute_gradients: Whether to compute the gradients of the
            observables.

    Returns:
        Either `stats` or, if `der_logs` is passed, a tuple of `stats` and `grad`:
            stats: A sequence of Stats object containing mean, variance,
                and MC diagonstics for each operator in `ops`.
            grad: A sequence of gradients of the expectation value of `op`,
                  as ndarray of shape `(psi.n_par,)`, for each `op` in `ops`.
    """

    from netket.operator import local_values as _local_values
    from ._C_netket import stats as nst

    psi = sampler.machine

    if not n_discard:
        n_discard = n_samples // 10

    # Burnout phase
    sampler.generate_samples(n_discard)
    # Generate samples
    samples = sampler.generate_samples(n_samples)

    if compute_gradients:
        der_logs = psi.der_log(samples)

    def estimate(op):
        lvs = _local_values(op, psi, samples)
        stats = nst.statistics(lvs)

        if compute_gradients:
            grad = nst.covariance_sv(lvs, der_logs)
            return stats, grad
        else:
            return stats

    return tree_map(estimate, ops)
