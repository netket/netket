from typing import Optional, Callable

from netket.utils.types import DType

from netket.hilbert import AbstractHilbert
from netket.operator import ContinousOperator

import jax.numpy as jnp


class PotentialEnergy(ContinousOperator):
    r"""Args:
    afun: The potential energy as function of x
    """

    def __init__(
        self, hilbert: AbstractHilbert, afun: Callable, dtype: Optional[DType] = float
    ):

        self._afun = afun
        self._dtype = dtype
        super().__init__(hilbert)

    @property
    def dtype(self) -> DType:
        return self._dtype

    def expect_kernel(self, logpsi, params, x, data):

        return jnp.sum(jnp.array(data) * self._afun(x))

    def _pack_arguments(self):
        return [1.0]
