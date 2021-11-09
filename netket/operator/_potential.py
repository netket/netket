from typing import Optional, Callable

from netket.utils.types import DType, PyTree, Array

from netket.hilbert import AbstractHilbert
from netket.operator import ContinousOperator

import jax.numpy as jnp


class PotentialEnergy(ContinousOperator):
    r"""
    Args:
        afun: The potential energy as function of x
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        afun: Callable,
        coefficient: float = 1.0,
        dtype: Optional[DType] = float,
    ):

        self._afun = afun
        self._dtype = dtype
        self.coefficient = jnp.array(coefficient)
        super().__init__(hilbert)

    @property
    def dtype(self) -> DType:
        return self._dtype

    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):

        return jnp.sum(jnp.array(data) * self._afun(x))

    def _pack_arguments(self):
        return self.coefficient

    def __repr__(self):
        return f"Potential(coefficient={self.coefficient}, function+{self._afun})"
