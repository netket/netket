# Copyright 2021 The NetKet Authors - All rights reserved.
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

import jax.numpy as jnp
import jax.scipy as jsp

from netket.jax import tree_ravel


def pinv_smooth(A, b, rcond=1e-14, rcond_smooth=1e-14, x0=None):
    r"""
    Solve the linear system by building a pseudo-inverse from the
    eigendecomposition obtained from :func:`jax.numpy.linalg.eigh`.

    The eigenvalues :math:`\lambda_i` smaller than
    :math:`r_\text{cond} \lambda_\text{max}` are truncated (where
    :math:`\lambda_\text{max}` is the largest eigenvalue).

    The eigenvalues are further smoothed with another filter, originally introduced in
    `Medvidovic, Sels arXiv:2212.11289 (2022) <https://arxiv.org/abs/2212.11289>`_,
    given by the following equation

    .. math::

        \tilde\lambda_i^{-1}=\frac{\lambda_i^{-1}}{1+\big(\epsilon\frac{\lambda_\text{max}}{\lambda_i}\big)^6}


    .. note::

        In general, we found that this custom implementation of
        the pseudo-inverse outperform
        jax's :func:`~jax.numpy.linalg.pinv`. This might be
        because :func:`~jax.numpy.linalg.pinv` internally calls
        :obj:`~jax.numpy.linalg.svd`, while this solver internally
        uses :obj:`~jax.numpy.linalg.eigh`.

        For that reason, we suggest you use this solver instead of
        :obj:`~netket.optimizer.solver.pinv`.


    Args:
        A: LinearOperator (matrix)
        b: vector or Pytree
        rcond : Cut-off ratio for small singular values of :code:`A`. For
            the purposes of rank determination, singular values are treated
            as zero if they are smaller than rcond times the largest
            singular value of :code:`A`.
        rcond_smooth : regularization parameter used with a similar effect to `rcond`
            but with a softer curve. See :math:`\epsilon` in the formula
            above.
    """
    del x0

    A = A.to_dense()
    b, unravel = tree_ravel(b)

    Σ, U = jnp.linalg.eigh(A)

    # Discard eigenvalues below numerical precision
    Σ_inv = jnp.where(jnp.abs(Σ / Σ[-1]) > rcond, jnp.reciprocal(Σ), 0.0)

    # Set regularizer for singular value cutoff
    regularizer = 1.0 / (1.0 + (rcond_smooth / jnp.abs(Σ / Σ[-1])) ** 6)

    Σ_inv = Σ_inv * regularizer

    x = U @ (Σ_inv * (U.conj().T @ b))

    return unravel(x), None


def pinv(A, b, rcond=1e-12, x0=None):
    """
    Solve the linear system using jax's implementation of the
    pseudo-inverse.

    Internally it calls :ref:`~jax.numpy.linalg.pinv` which
    uses a :ref:`~jax.numpy.linalg.svd` decomposition with
    the same value of **rcond**.

    .. note::

        In general, we found that our custom implementation of
        the pseudo-inverse
        :func:`netket.optimizer.solver.pinv_smooth` (which
        internally uses hermitian diagonaliation) outperform
        jax's :ref:`~jax.numpy.linalg.pinv`.

        For that reason, we suggest to use
        :func:`~netket.optimizer.solver.pinv_smooth` instead of
        :obj:`~netket.optimizer.solver.pinv`.


    The diagonal shift on the matrix can be 0 and the
    **rcond** variable can be used to truncate small
    eigenvalues.

    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        rcond: The condition number
    """
    del x0

    A = A.to_dense()
    b, unravel = tree_ravel(b)

    x, residuals, rank, s = jnp.linalg.lstsq(A, b, rcond=rcond)
    A_inv = jnp.linalg.pinv(A, rcond=rcond, hermitian=True)
    x = jnp.dot(A_inv, b)

    return unravel(x), None


def svd(A, b, rcond=None, x0=None):
    """
    Solve the linear system using Singular Value Decomposition.
    The diagonal shift on the matrix should be 0.

    Internally uses {ref}`jax.numpy.linalg.lstsq`.

    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        rcond: The condition number
    """
    del x0

    A = A.to_dense()
    b, unravel = tree_ravel(b)

    x, residuals, rank, s = jnp.linalg.lstsq(A, b, rcond=rcond)

    return unravel(x), (residuals, rank, s)


def cholesky(A, b, lower=False, x0=None):
    """
    Solve the linear system using a Cholesky Factorisation.
    The diagonal shift on the matrix should be 0.

    Internally uses {ref}`jax.numpy.linalg.cho_solve`.

    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        lower: if True uses the lower half of the A matrix
        x0: unused
    """

    del x0

    A = A.to_dense()
    b, unravel = tree_ravel(b)

    c, low = jsp.linalg.cho_factor(A, lower=lower)
    x = jsp.linalg.cho_solve((c, low), b)
    return unravel(x), None


def LU(A, b, trans=0, x0=None):
    """
    Solve the linear system using a LU Factorisation.
    The diagonal shift on the matrix should be 0.

    Internally uses {ref}`jax.numpy.linalg.lu_solve`.

    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        lower: if True uses the lower half of the A matrix
        x0: unused
    """

    del x0

    A = A.to_dense()
    b, unravel = tree_ravel(b)

    lu, piv = jsp.linalg.lu_factor(A)
    x = jsp.linalg.lu_solve((lu, piv), b, trans=0)
    return unravel(x), None


# I believe this internally uses a smarter/more efficient way to
# do cholesky
def solve(A, b, assume_a="pos", x0=None):
    """
    Solve the linear system.
    The diagonal shift on the matrix should be 0.

    Internally uses {ref}`jax.numpy.solve`.

    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        lower: if True uses the lower half of the A matrix
        x0: unused
    """
    del x0

    A = A.to_dense()
    b, unravel = tree_ravel(b)

    x = jsp.linalg.solve(A, b, assume_a="pos")
    return unravel(x), None
