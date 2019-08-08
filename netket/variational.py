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


def estimate_expectation(op, psi, mc_data, return_gradient=False):
    """
    estimate_expectation(op: AbstractOperator, psi: AbstractMachine, mc_data: MCResult, return_gradient: bool=True) -> Stats

    For a linear operator, computes a statistical estimate of the expectation value,
    variance, and optionally the gradient of the expectation value with respect to the
    variational parameters.

    The estimate is based on a Markov chain of configurations (`mc_data`), as
    returned from `compute_samples`.

    Args:
        op: Linear operator
        psi: Variational wavefunction
        mc_data: MC result obtained from `compute_samples`
        return_gradient (bool): Whether to compute and return the gradient of the
            expectation value of `op`. If True, `mc_data` needs to provide the
            _centered_ (i.e., with subtracted mean) logarithmic derivatives of the
            wavefunction, which can be obtained by calling
                compute_samples(..., der_logs="centered")

    Returns:
        Either `stats` or, if `return_gradient == True`, a tuple of `stats` and `grad`:
            stats: A Stats object containing mean, variance, and MC diagonstics for
                   the estimated expectation value of `op`.
            grad: The gradient of the expectation value of `op`,
                  as ndarray of shape `(psi.n_par,)`.
    """

    from ._C_netket import operator as nop
    from ._C_netket.stats import statistics

    local_values = nop.local_values(op, psi, mc_data.samples, mc_data.log_values)
    stats = statistics(local_values)

    if return_gradient:
        if mc_data.der_logs is None:
            raise ValueError(
                "`return_gradient=True` passed to `estimate expectation`, but "
                "`mc_data.der_logs` is not available. Call `compute_samples(...,"
                " der_logs='centered')`."
            )

        grad = covariance_sv(local_values, mc_data.der_logs)
        return stats, grad
    else:
        return stats
