from typing import Optional
from netket.utils.types import DType

from netket.hilbert import AbstractHilbert
from netket.operator import AbstractOperator

import jax
import jax.numpy as jnp


class DiffOperator(AbstractOperator):
    def __init__(self, hilbert: AbstractHilbert):
        r"""Args:
        hilbert: The Hilbert space
        """

        super().__init__(hilbert)

    def operator(self, logpsi, order):
        dlogpsi = jax.grad(logpsi)

        if order == 1:
            return lambda x: dlogpsi(x)

        elif order == 2:
            basis = jnp.ones(self.hilbert.size)
            dp_dx2 = lambda x: jnp.diag(jax.vmap(jax.linearize(dlogpsi, x)[1])(basis))
            return lambda x: dp_dx2(x) + dlogpsi(x) ** 2

        else:
            raise NotImplementedError
