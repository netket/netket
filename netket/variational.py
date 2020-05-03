import warnings

import numpy as _np

from ._vmc import Vmc as _Vmc, estimate_expectations
from .optimizer import SR as _SR


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
    sr_lsq_solver=None,
):

    if use_cholesky and sr_lsq_solver != "LLT":
        raise ValueError(
            "Inconsistent options specified: `use_cholesky && sr_lsq_solver != 'LLT'`."
        )

    if discarded_samples_on_init != 0:
        warnings.warn(
            "discarded_samples_on_init does not have any effect and should not be used",
            FutureWarning,
        )

    warnings.warn(
        "netket.variational.Vmc will be removed in version 3, use netket.Vmc instead",
        FutureWarning,
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
