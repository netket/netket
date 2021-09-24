from typing import Optional
from netket.utils.types import DType

from netket.hilbert import AbstractHilbert
from netket.operator import AbstractOperator

import jax
import jax.numpy as jnp


class KineticPotential(AbstractOperator):
    def __init__(self, hilbert: AbstractHilbert, afun, dtype: Optional[DType] = float):
        r"""Args:
        afun: The potential energy as function of x_in
        x_in (1Darray): A sample of particle positions
        """

        self.afun = afun
        self._dtype = dtype
        super().__init__(hilbert)

    @property
    def dtype(self) -> DType:
        return self._dtype

    def potential(self, x_in):
        r"""
        Args:
            x_in (1Darray): A sample of particle positions
        """

        return self.afun(x_in)

    def e_kin(self, log_psi, params, x_in):
        r"""Returns the local kinetic energy
        :math:`E_{kin} = -1/2 ( \sum_i (\log(\psi))'^2 + (\log(\psi))'' )`
        Args:
            log_psi: Log of the wavefunction
            params: parameters of the wavefucntion
            x_in (array): A sample of particle positions"""

        def logpsi_x(x):
            return log_psi(params, x)

        dlogpsi_x = jax.grad(logpsi_x)

        basis = jnp.eye(x_in.shape[0])

        y, f_jvp = jax.linearize(dlogpsi_x, x_in)
        dp_dx2 = jnp.diag(jax.vmap(f_jvp)(basis))
        """
        def hvp(df, tangents, x):
            return jax.jvp(df, (x,), (tangents,))[1]
    
        basis = jnp.eye(x_in.shape[0])
    
        dp_dx2 = jnp.trace(jax.vmap(hvp, in_axes=(None, 0, None), out_axes=0)(dlogpsi_x, basis, x_in))
        """
        dp_dx = dlogpsi_x(x_in) ** 2

        return -0.5 * jnp.sum(dp_dx2 + dp_dx)

    def total_energy(self, log_psi, params, x_in):
        return self.e_kin(log_psi, params, x_in) + self.potential(x_in)
