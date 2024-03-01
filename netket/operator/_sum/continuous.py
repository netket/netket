# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Union
from collections.abc import Hashable, Iterable
from netket.utils.types import PyTree, Array


import jax.numpy as jnp

from netket.utils import struct, HashableArray

from .._abstract_operator import AbstractOperator
from .._continuous_operator import ContinuousOperator

from .base import SumOperator


@struct.dataclass
class SumOperatorPyTree:
    """Internal class used to pass data from the operator to the jax kernel.

    This is used such that we can pass a PyTree containing some static data.
    We could avoid this if the operator itself was a pytree, but as this is not
    the case we need to pass as a separte object all fields that are used in
    the kernel.

    We could forego this, but then the kernel could not be marked as
    @staticmethod and we would recompile every time we construct a new operator,
    even if it is identical
    """

    ops: tuple[ContinuousOperator, ...] = struct.field(pytree_node=False)
    coeffs: Array
    op_data: tuple[PyTree, ...]


class SumContinuousOperator(SumOperator, ContinuousOperator):
    def __init__(
        self,
        *operators: AbstractOperator,
        coefficients: Union[float, Iterable[float]] = 1.0,
        dtype=None,
    ):
        if not all(isinstance(op, AbstractOperator) for op in operators):
            raise TypeError(
                "Arguments to SumOperator must all be subtypes of "
                "AbstractOperator. However the types are:\n\n"
                f"{list(type(op) for op in operators)}\n"
            )
        super().__init__(
            operators, operators[0].hilbert, coefficients=coefficients, dtype=dtype
        )

    def __add__(self, other):
        if isinstance(other, SumOperator):
            ops = self.operators + other.operators
            coeffs = jnp.concatenate([self.coefficients, other.coefficents])
            dtype = self.dtype if self.dtype == other.dtype else None
        else:
            ops = (*self.operators, other)
            coeffs = jnp.concatenate([self.coefficients, jnp.array([1.0])])
            dtype = self.dtype if self.dtype == other.dtype else None

        return SumContinuousOperator(*ops, coefficients=coeffs, dtype=dtype)

    @staticmethod
    def _expect_kernel(
        logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):
        result = [
            data.coeffs[i] * op._expect_kernel(logpsi, params, x, op_data)
            for i, (op, op_data) in enumerate(zip(data.ops, data.op_data))
        ]

        return sum(result)

    def _pack_arguments(self) -> SumOperatorPyTree:
        return SumOperatorPyTree(
            self.operators,
            self.coefficients,
            tuple(op._pack_arguments() for op in self.operators),
        )

    @property
    def _attrs(self) -> tuple[Hashable, ...]:
        if self.__attrs is None:
            self.__attrs = (
                self.hilbert,
                self.operators,
                HashableArray(self.coefficients),
                self.dtype,
            )
        return self.__attrs
