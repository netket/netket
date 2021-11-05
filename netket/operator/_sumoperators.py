from typing import Optional, List

from netket.utils.types import DType

from netket.hilbert import AbstractHilbert
from netket.operator import ContinousOperator

import jax.numpy as jnp

class SumOperator(ContinousOperator):

    def __init__(self, hilbert: AbstractHilbert, operators: List, dtype: Optional[DType] = float):

        self._ops = operators

        super().__init__(hilbert)

    def expect_kernel(self, logpsi, params, x, data):
        result = [op.expect_kernel(logpsi, params, x, data) for op in self._ops]
        return jnp.sum(jnp.array(result))

    def pack_data(self):
        return [op.pack_data() for op in self._ops]
