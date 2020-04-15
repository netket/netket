from ._vmc import Vmc as _Vmc
from ._C_netket.optimizer import SR as _SR
import warnings
from netket.vmc_common import tree_map


def Vmc(
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

    if use_cholesky and sr_lsq_solver != "LLT":
        raise ValueError(
            "Inconsistent options specified: `use_cholesky && sr_lsq_solver != 'LLT'`."
        )

    if discarded_samples_on_init != 0:
        warnings.warn(
            "discarded_samples_on_init does not have any effect and should not be used",
            DeprecationWarning,
        )

    warnings.warn(
        "netket.variational.Vmc will be deprecated in version 3, use netket.Vmc instead",
        PendingDeprecationWarning,
    )

    if method == "Gd":
        return _Vmc(
            hamiltonian=hamiltonian,
            sampler=sampler,
            optimizer=optimizer,
            n_samples=n_samples,
            n_discard=discarded_samples,
            sr=None,
        )
    elif method == "Sr":
        sr = _SR(
            lsq_solver=sr_lsq_solver,
            diag_shift=diag_shift,
            use_iterative=use_iterative,
            is_holomorphic=sampler.machine.is_holomorphic,
        )
        return _Vmc(
            hamiltonian=hamiltonian,
            sampler=sampler,
            optimizer=optimizer,
            n_samples=n_samples,
            n_discard=discarded_samples,
            sr=sr,
        )
    else:
        raise ValueError("Allowed method options are Gd and Sr")


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
