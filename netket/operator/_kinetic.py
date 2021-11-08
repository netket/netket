from typing import Optional, Union

from netket.utils.types import DType, PyTree

from netket.hilbert import AbstractHilbert
from netket.operator import ContinousOperator

import jax
import jax.numpy as jnp


class KineticEnergy(ContinousOperator):
    r"""Returns the local kinetic energy (hbar = 1)
                :math:`E_{kin} = -1/2 ( \sum_i \frac{1}{m_i} (\log(\psi))'^2 + (\log(\psi))'' )`
    Args:
    mass: float if all masses are the same, list indicating the mass of each particle otherwise
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        mass: Union[float, list],
        dtype: Optional[DType] = float,
    ):

        self._mass = mass
        self._dtype = dtype

        super().__init__(hilbert)

    @property
    def dtype(self) -> DType:
        return self._dtype

    def expect_kernel(self, logpsi, params, x, data):
        def logpsi_x(x):
            return logpsi(params, x)

        dlogpsi_x = jax.grad(logpsi_x)

        basis = jnp.eye(x.shape[0])

        y, f_jvp = jax.linearize(dlogpsi_x, x)
        dp_dx2 = jnp.diag(jax.vmap(f_jvp)(basis))

        dp_dx = dlogpsi_x(x) ** 2

        return -0.5 * jnp.sum(1.0 / jnp.array(data) * (dp_dx2 + dp_dx))

    def _pack_arguments(self) -> PyTree:
        if isinstance(self._mass, list):
            return self._mass

        return self.hilbert.size * [self._mass]
