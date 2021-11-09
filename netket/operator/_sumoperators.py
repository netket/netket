from typing import Union, List, Optional, Callable

from netket.utils.types import DType, PyTree, Array

import functools

from netket.operator import ContinousOperator

import jax.numpy as jnp


class SumOperator(ContinousOperator):
    def __init__(
        self,
        *operators: List,
        coefficients: Union[float, List[float]] = 1.0,
        dtype: Optional[DType] = None,
    ):
        r"""
        Args:
            hilbert: The underlying Hilbert space on which the operators are defined
            operators: A list of ContinousOperator objects
            coeff: A coefficient for each ContinousOperator object
        """

        hil = [op.hilbert for op in operators]
        if not all(_ == hil[0] for _ in hil):
            raise NotImplementedError(
                "Cannot add operators on different hilbert spaces"
            )

        self._ops = operators
        self._coeff = coefficients

        if dtype is None:
            dtype = functools.reduce(
                lambda dt, op: jnp.promote_types(dt, op.dtype), operators, float
            )
        self._dtype = dtype

        super().__init__(hil[0])

    @property
    def dtype(self) -> DType:
        return self._dtype

    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):
        result = [
            op._expect_kernel(logpsi, params, x, data[i])
            for i, op in enumerate(self._ops)
        ]
        return jnp.sum(jnp.array(result))

    def _pack_arguments(self):
        return [self._coeff * jnp.array(op._pack_arguments()) for op in self._ops]
