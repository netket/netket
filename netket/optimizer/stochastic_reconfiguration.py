from functools import singledispatch
from . import numpy


@singledispatch
def SR(
    machine,
    lsq_solver=None,
    diag_shift=0.01,
    use_iterative=True,
    svd_threshold=None,
    sparse_tol=None,
    sparse_maxiter=None,
):
    return numpy.SR(
        machine,
        lsq_solver,
        diag_shift,
        use_iterative,
        svd_threshold,
        sparse_tol,
        sparse_maxiter,
    )
