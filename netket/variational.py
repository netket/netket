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


def estimate_expectation(op, psi, samples, log_values, der_logs=None):
    """
    estimate_expectation(op: AbstractOperator, psi: AbstractMachine, mc_data: MCResult, return_gradient: bool=True) -> Stats

    For a linear operator, computes a statistical estimate of the expectation value,
    variance, and optionally the gradient of the expectation value with respect to the
    variational parameters.

    The estimate is based on a Markov chain of configurations obtained from
    `netket.variational.compute_samples`.

    Args:
        op: Linear operator
        psi: Variational wavefunction
        samples: A matrix (or a rank-3 tensor) of visible
            configurations. If it is a matrix, each row of the matrix
            must correspond to a visible configuration.  `samples` is a
            rank-3 tensor, its shape should be `(N, M, #visible)` where
            `N` is the number of samples, `M` is the number of Markov
            Chains, and `#visible` is the number of visible units.
        log_values: Corresponding values of the logarithm of the
            wavefunction. If `samples` is a `(N, #visible)` matrix, then
            `log_values` should be a vector of `N` complex numbers. If
            `samples` is a rank-3 tensor, then the shape of `log_values`
            should be `(N, M)`.
        der_logs: Logarithmic derivatives of the wavefunction
            If `samples` is a `(N, #visible)` matrix, `der_logs` should have
            shape `(N, psi.n_par)`. If `samples` is a rank-3 tensor, the shape
            of `der_logs` should be `(N, M, psi.n_par)`.
            This argument is optional; if it is not passed, the gradient will
            not be computed.

    Returns:
        Either `stats` or, if `der_logs` is passed, a tuple of `stats` and `grad`:
            stats: A Stats object containing mean, variance, and MC diagonstics for
                   the estimated expectation value of `op`.
            grad: The gradient of the expectation value of `op`,
                  as ndarray of shape `(psi.n_par,)`.
    """

    from ._C_netket import operator as nop
    from ._C_netket.stats import covariance_sv, statistics

    local_values = nop.local_values(op, psi, samples, log_values)
    stats = statistics(local_values)

    if der_logs is not None:
        grad = covariance_sv(local_values, der_logs)
        return stats, grad
    else:
        return stats
