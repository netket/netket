# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools as _itertools

from . import _core
from ._C_netket.exact import *


def _ExactTimePropagation_iter(self, dt, n_iter=None):
    """
    iter(self: ExactTimePropagation, dt: float, n_iter: int=None) -> int

    Returns a generator which advances the time evolution by dt,
    yielding after every step.

    Args:
        dt (float): The size of the time step.
        n_iter (int=None): The number of steps or None, for no limit.

    Yields:
        int: The current step.
    """
    for i in _itertools.count():
        if n_iter and i >= n_iter:
            return
        self.advance(dt)
        yield i


class EdResult(object):
    def __init__(self, eigenvalues, eigenvectors):
        # NOTE: These conversions are required because our old C++ code stored
        # eigenvalues and eigenvectors as Python lists :(
        self._eigenvalues = eigenvalues.tolist()
        self._eigenvectors = (
            [eigenvectors[:, i] for i in range(eigenvectors.shape[1])]
            if eigenvectors is not None
            else []
        )

    @property
    def eigenvalues(self):
        r"""Eigenvalues of the Hamiltonian.
        """
        return self._eigenvalues

    @property
    def eigenvectors(self):
        r"""Eigenvectors of the Hamiltonian.
        """
        return self._eigenvectors

    def mean(self, operator, which):
        import numpy

        x = self._eigenvectors[which]
        return numpy.vdot(x, operator(x))


def lanczos_ed(
    operator,
    matrix_free=False,
    first_n=1,
    max_iter=1000,
    seed=None,
    precision=1e-14,
    compute_eigenvectors=False,
):
    r"""Computes `first_n` smallest eigenvalues and, optionally, eigenvectors
    of a Hermitian operator using the Lanczos method.

    Args:
        operator: The operator to diagnolize.
        matrix_free: If true, matrix elements are computed on the fly.
            Otherwise, the operator is first converted to a sparse matrix.
        first_n: The number of eigenvalues to compute.
        max_iter: The maximum number of iterations.
        seed: **Ignored**. Accepted for backward compatibility only.
        precision: The precision to which the eigenvalues will be
            computed.
        compute_eigenvectors: Whether or not to return the
            eigenvectors of the operator. With ARPACK, not requiring the
            eigenvectors has almost no performance benefits.


    Examples:
        Testing the number of eigenvalues saved when solving a simple
        1D Ising problem.

        ```python
        >>> import netket as nk
        >>> hilbert = nk.hilbert.Spin(
        ...     nk.graph.Hypercube(length=8, n_dim=1, pbc=True), s=0.5)
        >>> hamiltonian = nk.operator.Ising(h=1.0, hilbert=hilbert)
        >>> r = nk.exact.lanczos_ed(
        ...     hamiltonian, first_n=3, compute_eigenvectors=True)
        >>> r.eigenvalues
        [-10.251661790966047, -10.054678984251746, -8.690939214837037]
        ```

    """
    from scipy.sparse.linalg import eigsh

    result = eigsh(
        operator.to_linear_operator() if matrix_free else operator.to_sparse(),
        k=first_n,
        which="SA",
        maxiter=max_iter,
        tol=precision,
        return_eigenvectors=compute_eigenvectors,
    )
    if compute_eigenvectors:
        return EdResult(*result)
    return EdResult(result, None)


def full_ed(operator, first_n=1, compute_eigenvectors=False):
    r"""Computes `first_n` smallest eigenvalues and, optionally, eigenvectors
    of a Hermitian operator by full diagonalization.

    Args:
        operator: Operator to diagnolize.
        first_n: (Deprecated) Number of eigenvalues to compute.
            This has no performance impact, as full_ed will compute all
            eigenvalues anyway.
        compute_eigenvectors: Whether or not to return the
            eigenvectors of the operator.

    Examples:
        Testing the numer of eigenvalues saved when solving a simple
        1D Ising problem.

        ```python
        >>> import netket as nk
        >>> hilbert = nk.hilbert.Spin(
        ...     nk.graph.Hypercube(length=8, n_dim=1, pbc=True), s=0.5)
        >>> hamiltonian = nk.operator.Ising(h=1.0, hilbert=hilbert)
        >>> r = nk.exact.lanczos_ed(
        ...     hamiltonian, first_n=3, compute_eigenvectors=True)
        >>> len(r.eigenvalues)
        3
        ```
    """
    from numpy.linalg import eigh, eigvalsh

    dense_op = operator.to_dense()

    if not (1 <= first_n < dense_op.shape[0]):
        raise ValueError("first_n must be in range 1..dim(operator)")

    if compute_eigenvectors:
        w, v = eigh(dense_op)
        return EdResult(w[:first_n], v[:, :first_n])
    else:
        w = eigvalsh(dense_op)
        return EdResult(w[:first_n], None)


ExactTimePropagation.iter = _ExactTimePropagation_iter


@_core.deprecated(
    "`ImagTimePropagation` is deprecated. Please use "
    '`ExactTimePropagation(..., propagation_type="imaginary")` instead.'
)
def ImagTimePropagation(*args, **kwargs):
    """
    Returns `ExactTimePropagation(..., propagation_type="imaginary")` for
    backwards compatibility.

    Deprecated (NetKet 2.0): Use `ExactTimePropagation` directly.
    """
    kwargs["propagation_type"] = "imaginary"
    return ExactTimePropagation(*args, **kwargs)
