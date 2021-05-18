import jax
import jax.numpy as jnp
import jax.scipy as jsp

from netket.jax import tree_ravel


def svd(A, b, rcond=None, x0=None):
    """
    Solve the linear system using Singular Value Decomposition.
    The diagonal shift on the matrix should be 0.

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
    del x0

    A = A.to_dense()
    b, unravel = tree_ravel(b)

    c, low = jsp.linalg.cho_factor(A, lower=lower)
    x = jsp.linalg.cho_solve((c, low), b)
    return unravel(x), None


def LU(A, b, trans=0, x0=None):
    del x0

    A = A.to_dense()
    b, unravel = tree_ravel(b)

    lu, piv = jsp.linalg.lu_factor(A)
    x = jsp.linalg.lu_solve((lu, piv), b, trans=0)
    return unravel(x), None


# I believe this internally uses a smarter/more efficient way to
# do cholesky
def solve(A, b, sym_pos=True, x0=None):
    del x0

    A = A.to_dense()
    b, unravel = tree_ravel(b)

    x = jsp.linalg.solve(A, b, sym_pos=sym_pos)
    return unravel(x), None
