from typing import Optional, Callable, Union

from netket.utils.types import DType

from netket.hilbert import AbstractHilbert
from netket.operator import ContinousOperator

import jax.numpy as jnp


class PotentialEnergy(ContinousOperator):
    def __init__(
        self, hilbert: AbstractHilbert, afun: Callable, dtype: Optional[DType] = float
    ):
        r"""Args:
        afun: The potential energy as function of x_in
        x_in (1Darray): A sample of particle positions
        """

        self._afun = afun
        self._dtype = dtype
        super().__init__(hilbert)

    @property
    def dtype(self) -> DType:
        return self._dtype

    def expect_kernel(self, logpsi, params, x_in, data):
        r"""
        Args:
            x_in (1Darray): A sample of particle positions
        """

        return jnp.sum(jnp.array(data) * self._afun(x_in))

    def pack_data(self):
        return [1.0]
