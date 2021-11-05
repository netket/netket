from typing import Optional, Union

from netket.utils.types import DType, PyTree

from netket.hilbert import AbstractHilbert
from netket.operator import ContinousOperator

import jax
import jax.numpy as jnp


class KineticEnergy(ContinousOperator):
    def __init__(
        self,
        hilbert: AbstractHilbert,
        mass: Union[float, list],
        dtype: Optional[DType] = float,
    ):
        r"""Args:
        mass: int if all masses are the same, array indicating the mass of each particle otherwise
        """

        self._dtype = dtype
        self._mass = mass
        super().__init__(hilbert)

    @property
    def dtype(self) -> DType:
        return self._dtype

    def expect_kernel(self, logpsi, params, x_in, data):
        r"""
        Args:
            x_in (1Darray): A sample of particle positions
        """

        r"""Returns the local kinetic energy
                    :math:`E_{kin} = -1/2 ( \sum_i (\log(\psi))'^2 + (\log(\psi))'' )`
                    Args:
                        log_psi: Log of the wavefunction
                        params: parameters of the wavefucntion
                        x_in (array): A sample of particle positions"""

        def logpsi_x(x):
            return logpsi(params, x)

        dlogpsi_x = jax.grad(logpsi_x)

        basis = jnp.eye(x_in.shape[0])

        y, f_jvp = jax.linearize(dlogpsi_x, x_in)
        dp_dx2 = jnp.diag(jax.vmap(f_jvp)(basis))

        dp_dx = dlogpsi_x(x_in) ** 2

        return -0.5 * jnp.sum(1.0 / jnp.array(data) * (dp_dx2 + dp_dx))

    def pack_data(self) -> PyTree:
        if isinstance(self._mass, list):
            return self._mass

        return self.hilbert.size * [self._mass]

        """
        def hvp(df, tangents, x):
            return jax.jvp(df, (x,), (tangents,))[1]

        basis = jnp.eye(x_in.shape[0])

        dp_dx2 = jnp.trace(jax.vmap(hvp, in_axes=(None, 0, None), out_axes=0)(dlogpsi_x, basis, x_in))
        """
