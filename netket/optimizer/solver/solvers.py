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
def solve(A, b, sym_pos=True, x0=None):
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

    x = jsp.linalg.solve(A, b, sym_pos=sym_pos)
    return unravel(x), None
