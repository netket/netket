from typing import Optional

from netket.utils.types import DType

from netket.hilbert import AbstractHilbert
from netket.operator import AbstractOperator

class ContinousOperator(AbstractOperator):
    def __init__(self, hilbert: AbstractHilbert, dtype: Optional[DType] = float):

        self._dtype = dtype
        super().__init__(hilbert)

    @property
    def dtype(self) -> DType:
        return self._dtype

    def expect_kernel(self, logpsi, params, x_in, data):
        return NotImplementedError

    def pack_data(self):
        return NotImplementedError

    def __add__(self, other):
        if isinstance(self, ContinousOperator) and isinstance(other, ContinousOperator):
            from netket.operator import SumOperator
            return SumOperator(self.hilbert, [self, other], self.dtype)
        else:
            NotImplementedError

