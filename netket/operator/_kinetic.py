from typing import Optional

from netket.utils.types import DType, PyTree

from netket.hilbert import AbstractHilbert
from netket.operator import ContinousOperator

import jax
import jax.numpy as jnp


class KineticEnergy(ContinousOperator):
    def __init__(self, hilbert: AbstractHilbert, mass: PyTree, dtype: Optional[DType] = float):
        r"""Args:
        afun: The potential energy as function of x_in
        x_in (1Darray): A sample of particle positions
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

        return -0.5 * jnp.sum(1./jnp.array(data)[0,:] * (dp_dx2 + dp_dx))

    def pack_data(self) -> PyTree:

        return self._mass

        """
        def hvp(df, tangents, x):
            return jax.jvp(df, (x,), (tangents,))[1]

        basis = jnp.eye(x_in.shape[0])

        dp_dx2 = jnp.trace(jax.vmap(hvp, in_axes=(None, 0, None), out_axes=0)(dlogpsi_x, basis, x_in))
        """
