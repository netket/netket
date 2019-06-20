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

import warnings
import functools
import itertools
import inspect

from ._C_netket.exact import *

# NOTE: If more modules end up requiring this functionality, we can create a
# `netket._core` module and move `deprecated` there.
def deprecated(reason=None):
    """
    This is a decorator which can be used to mark functions as deprecated. It
    will result in a warning being emitted when the function is used.
    """

    def decorator(func):
        object_type = "class" if inspect.isclass(func) else "function"
        message = "Call to deprecated {} {!r}".format(object_type, func.__name__)
        if reason is not None:
            message += " ({})".format(reason)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, category=FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _ImagTimePropagation_iter(self, dt, n_iter=None):
    """
    iter(self: ImagTimePropagation, dt: float, n_iter: int=None) -> int

    Returns a generator which advances the time evolution by dt,
    yielding after every step.

    Args:
        dt (float): The size of the time step.
        n_iter (int=None): The number of steps or None, for no limit.

    Yields:
        int: The current step.
    """
    for i in itertools.count():
        if n_iter and i >= n_iter:
            return
        self.advance(dt)
        yield i


ImagTimePropagation.iter = _ImagTimePropagation_iter


@deprecated()
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


@deprecated(
    "use 'netket.Operator.to_linear_operator' to convert an operator "
    "to a 'scipy.sparse.linalg.LinearOperator' and then call "
    "'scipy.sparse.linalg.eigsh' directly"
)
def lanczos_ed(
    operator,
    matrix_free=False,
    first_n=1,
    max_iter=1000,
    seed=None,
    precision=1e-14,
    compute_eigenvectors=False,
):
    r"""Uses Lanczos algorithm to diagonalize the operator.

    Args:
        operator: The operator to diagnolize.
        matrix_free: If true, matrix elements are computed on the fly.
            Otherwise, the operator is first converted to a sparse matrix.
        first_n: The number of eigenvalues to compute.
        max_iter: The maximum number of iterations.
        seed: **Ignored**. Accepted for backward compatibility only.
        precision: The precision to which the eigenvalues will be
            computed.
        compute_eigenvectors: Whether or not to compute the
            eigenvectors of the operator.


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


@deprecated(
    "use 'netket.Operator.to_linear_operator' to convert an operator "
    "to a 'scipy.sparse.linalg.LinearOperator' and then call "
    "'scipy.sparse.linalg.eigsh' directly"
)
def full_ed(operator, first_n=1, compute_eigenvectors=False):
    r"""Diagonalizes the operator.

    Args:
        operator: Operator to diagnolize.
        first_n: Number of eigenvalues to compute.
        compute_eigenvectors: Whether or not to compute the
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
    return lanczos_ed(
        operator,
        matrix_free=False,
        first_n=first_n,
        compute_eigenvectors=compute_eigenvectors,
    )
