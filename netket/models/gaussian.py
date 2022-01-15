import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import normal

from netket.utils.types import DType, Array, NNInitFunc


class Gaussian(nn.Module):
    r"""
    Multivariate Gaussain function with mean 0 and parametrised covariance matrix
    :math:`\Sigma_{ij}`.

    The wavefunction is given by the formula: :math:`\Psi(x) = \exp(\sum_{ij} x_i \Sigma_{ij} x_j)`.
    The (positive definite) :math:`\Sigma_{ij} = AA^T` matrix is stored as
    non-positive definite matrix A.
    """

    dtype: DType = jnp.float64
    """The dtype of the weights."""
    kernel_init: NNInitFunc = normal(stddev=1.0)
    """Initializer for the weights."""

    @nn.compact
    def __call__(self, x_in: Array):
        nv = x_in.shape[-1]

        dtype = jnp.promote_types(x_in.dtype, self.dtype)
        x_in = jnp.asarray(x_in, dtype=dtype)

        kernel = self.param("kernel", self.kernel_init, (nv, nv), self.dtype)

        kernel = jnp.dot(kernel.T, kernel)

        # print(kernel)
        y = -0.5 * jnp.einsum("...i,ij,...j", x_in, kernel, x_in)

        return y
