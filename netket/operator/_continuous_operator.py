from typing import Optional

from netket.utils.types import DType

from netket.hilbert import AbstractHilbert
from netket.operator import AbstractOperator


class ContinousOperator(AbstractOperator):
    r"""This class is the base class for operators defined on a
    continuous space Hilbert space. Users interested in implementing new
    quantum Operators for continuous Hilbert spaces should derive
    their own class from this class
    """

    def __init__(self, hilbert: AbstractHilbert, dtype: Optional[DType] = float):

        self._dtype = dtype
        super().__init__(hilbert)

    @property
    def dtype(self) -> DType:
        return self._dtype

    def expect_kernel(self, logpsi, params, x_in, data):
        r"""In this method the action of the operator on a given quantum state
        logpsi at a given congfiguration x_in is defined."""
        raise NotImplementedError

    def pack_data(self):
        r"""This methods makes it possible to give coefficients to the expect_kernel method above.
        For example for the kinetic energy this method would return the masses of the
        individual particles."""
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(self, ContinousOperator) and isinstance(other, ContinousOperator):
            from netket.operator import SumOperator

            return SumOperator(self.hilbert, [self, other], 1.0, self.dtype)
        else:
            return NotImplementedError

    def __rmul__(self, other):
        if isinstance(self, ContinousOperator) and isinstance(other, float):
            from netket.operator import SumOperator

            return SumOperator(self.hilbert, [self], other, self.dtype)
        else:
            return NotImplementedError

    def __mul__(self, other):
        if isinstance(self, ContinousOperator) and isinstance(other, float):
            from netket.operator import SumOperator

            return SumOperator(self.hilbert, [self], other, self.dtype)
        else:
            return NotImplementedError
