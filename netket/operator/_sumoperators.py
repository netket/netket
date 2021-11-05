from typing import Optional, List

from netket.utils.types import DType

from netket.hilbert import AbstractHilbert
from netket.operator import ContinousOperator

import jax.numpy as jnp


class SumOperator(ContinousOperator):
    def __init__(
        self,
        hilbert: AbstractHilbert,
        operators: List,
        coeff: float,
        dtype: Optional[DType] = float,
    ):

        self._ops = operators
        self._coeff = coeff
        super().__init__(hilbert)

    @property
    def dtype(self) -> DType:
        return self._dtype

    def expect_kernel(self, logpsi, params, x, data):
        result = [
            op.expect_kernel(logpsi, params, x, data[i])
            for i, op in enumerate(self._ops)
        ]
        return jnp.sum(jnp.array(result))

    def pack_data(self):
        return [jnp.array(self._coeff) * jnp.array(op.pack_data()) for op in self._ops]
