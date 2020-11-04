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

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab

from . import _core
from ._exact_dynamics import PyExactTimePropagation


class EdResult(object):
    def __init__(self, eigenvalues, eigenvectors):
        # NOTE: These conversions are required because our old C++ code stored
        # eigenvalues and eigenvectors as Python lists :(
        self._eigenvalues = eigenvalues.tolist()
        self._eigenvectors = (
            [np.asarray(eigenvectors[:, i]) for i in range(eigenvectors.shape[1])]
            if eigenvectors is not None
            else []
        )

    @property
    def eigenvalues(self):
        r"""Eigenvalues of the Hamiltonian."""
        return self._eigenvalues

    @property
    def eigenvectors(self):
        r"""Eigenvectors of the Hamiltonian."""
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


def full_ed(operator, compute_eigenvectors=False):
    r"""Computes `first_n` smallest eigenvalues and, optionally, eigenvectors
    of a Hermitian operator by full diagonalization.

    Args:
        operator: Operator to diagnolize.
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
        ...     hamiltonian, compute_eigenvectors=True)
        >>> len(r.eigenvalues)
        3
        ```
    """
    from numpy.linalg import eigh, eigvalsh

    dense_op = operator.to_dense()

    if compute_eigenvectors:
        w, v = eigh(dense_op)
        return EdResult(w, v)
    else:
        w = eigvalsh(dense_op)
        return EdResult(w, None)


def steady_state(lindblad, sparse=False, method="ed", rho0=None, **kwargs):
    r"""Computes the numerically exact steady-state of a lindblad master equation.
    The computation is performed either through the exact diagonalization of the
    hermitian L^\dagger L matrix, or by means of an iterative solver (bicgstabl)
    targeting the solution of the non-hermitian system L\rho = 0 && \Tr[\rho] = 1.

    Note that for systems with 7 or more sites it is usually computationally impossible
    to build the full lindblad operator and therefore only `iterative` will work.

    Note that for systems with hilbert spaces with dimensions above 40k, tol
    should be set to a lower value if the steady state has non-trivial correlations.

    Args:
        lindblad: The lindbladian encoding the master equation.
        sparse: Whever to use sparse matrices (default: False)
        method: 'ed' (exact diagonalization) or 'iterative' (iterative bicgstabl)
        rho0: starting density matrix for the iterative diagonalization (default: None)
        kwargs...: additional kwargs passed to bicgstabl

    Optional args for iterative:
        For full docs please consult SciPy documentation at
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.bicgstab.html

        maxiter: maximum number of iterations for the iterative solver (default: None)
        tol: The precision for the calculation (default: 1e-05)
        callback: User-supplied function to call after each iteration. It is called as callback(xk),
                  where xk is the current solution vector

    """
    from numpy import sqrt, matrix

    M = lindblad.hilbert.hilbert_physical.n_states

    if method == "ed":
        if not sparse:
            from numpy.linalg import eigh

            lind_mat = matrix(lindblad.to_dense())

            ldagl = lind_mat.H * lind_mat
            w, v = eigh(ldagl)

        else:
            from scipy.sparse.linalg import eigsh

            lind_mat = lindblad.to_sparse()
            ldagl = lind_mat.H * lind_mat

            w, v = eigsh(ldagl, which="SM", k=2)

        print("Minimum eigenvalue is: ", w[0])
        rho = matrix(v[:, 0].reshape((M, M)))
        rho = rho / rho.trace()

    elif method == "iterative":

        iHnh = -1j * lindblad.get_effective_hamiltonian()
        if sparse:
            iHnh = iHnh.to_sparse()
            J_ops = [j.to_sparse() for j in lindblad.jump_ops]
        else:
            iHnh = iHnh.to_dense()
            J_ops = [j.to_dense() for j in lindblad.jump_ops]

        # This function defines the product Liouvillian x densitymatrix, without
        # constructing the full density matrix (passed as a vector M^2).

        # An extra row is added at the bottom of the therefore M^2+1 long array,
        # with the trace of the density matrix. This is needed to enforce the
        # trace-1 condition.

        # The logic behind the use of Hnh_dag_ and Hnh_ is derived from the
        # convention adopted in local_liouvillian.cc, and inspired from reference
        # arXiv:1504.05266
        def matvec(rho_vec):
            rho = rho_vec[:-1].reshape((M, M))

            out = np.zeros((M ** 2 + 1), dtype="complex128")
            drho = out[:-1].reshape((M, M))

            drho += rho @ iHnh + iHnh.conj().T @ rho
            for J in J_ops:
                drho += (J @ rho) @ J.conj().T

            out[-1] = rho.trace()
            return out

        L = LinearOperator((M ** 2 + 1, M ** 2 + 1), matvec=matvec)

        # Initial density matrix ( + trace condition)
        Lrho_start = np.zeros((M ** 2 + 1), dtype="complex128")
        if rho0 is None:
            Lrho_start[0] = 1.0
            Lrho_start[-1] = 1.0
        else:
            Lrho_start[:-1] = rho0.reshape(-1)
            Lrho_start[-1] = rho0.trace()

        # Target residual (everything 0 and trace 1)
        Lrho_target = np.zeros((M ** 2 + 1), dtype="complex128")
        Lrho_target[-1] = 1.0

        # Iterative solver
        print("Starting iterative solver...")
        res, info = bicgstab(L, Lrho_target, x0=Lrho_start, **kwargs)

        rho = res[1:].reshape((M, M))
        if info == 0:
            print("Converged trace residual is ", res[-1])
        elif info > 0:
            print(
                "Failed to converge after ", info, " ( traceresidual is ", res[-1], " )"
            )
        elif info < 0:
            print("An error occured: ", info)

    else:
        raise ValueError("method must be 'ed'")

    return rho
