from ._C_netket.variational import *

import itertools


def _Vmc_iter(self, n_iter=None, step_size=1):
    """
    iter(self: Vmc, n_iter: int=None, step_size: int=1) -> int

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


Vmc.iter = _Vmc_iter


# Higher-level VMC functions:


def estimate_expectations(
    ops, sampler, n_samples, n_discard=None, compute_gradients=False
):
    """
    estimate_expectation(op: AbstractOperator, psi: AbstractMachine, mc_data: MCResult, return_gradient: bool=True) -> Stats

    For a sequence of linear operators, computes a statistical estimate of the
    respective expectation values, variances, and optionally gradients of the
    expectation values with respect to the variational parameters.

    The estimate is based on a Markov chain of `n_samples` configurations
    obtained from `netket.variational.compute_samples`.

    Args:
        ops: Sequence of linear operators
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

    from ._C_netket import operator as nop
    from ._C_netket import stats as nst
    from ._C_netket.sampler import compute_samples

    psi = sampler.machine

    if not n_discard:
        n_discard = n_samples // 10

    samples, log_values = compute_samples(sampler, n_samples, n_discard)

    local_values = [nop.local_values(op, psi, samples, log_values) for op in ops]
    stats = [nst.statistics(lv) for lv in local_values]

    if compute_gradients:
        der_logs = psi.der_log(samples)
        grad = [nst.covariance_sv(lv, der_logs) for lv in local_values]
        return stats, grad
    else:
        return stats
