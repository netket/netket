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

from netket.utils.numbers import is_scalar
from netket.utils.types import DType, PyTree, Array

from netket.jax import canonicalize_dtypes
from netket.operator import ContinuousOperator
from netket.utils import struct, HashableArray

import jax.numpy as jnp


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


def _flatten_sumoperators(operators: Iterable[ContinuousOperator], coefficients: Array):
    """Flatten sumoperators inside of operators."""
    new_operators = []
    new_coeffs = []
    for op, c in zip(operators, coefficients):
        if isinstance(op, SumOperator):
            new_operators.extend(op.operators)
            new_coeffs.extend(c * op.coefficients)
        else:
            new_operators.append(op)
            new_coeffs.append(c)
    return new_operators, new_coeffs


class SumOperator(ContinuousOperator):
    r"""This class implements the action of the _expect_kernel()-method of
    ContinuousOperator for a sum of ContinuousOperator objects.
    """

    def __init__(
        self,
        *operators: tuple[ContinuousOperator, ...],
        coefficients: Union[float, Iterable[float]] = 1.0,
        dtype: Optional[DType] = None,
    ):
        r"""
        Returns the action of a sum of local operators.
        Args:
            operators: A list of ContinuousOperator objects
            coefficients: A coefficient for each ContinuousOperator object
            dtype: Data type of the coefficients
        """
        hi_spaces = [op.hilbert for op in operators]
        if not all(hi == hi_spaces[0] for hi in hi_spaces):
            raise NotImplementedError(
                "Cannot add operators on different hilbert spaces"
            )

        if is_scalar(coefficients):
            coefficients = [coefficients for _ in operators]

        if len(operators) != len(coefficients):
            raise AssertionError("Each operator needs a coefficient")

        operators, coefficients = _flatten_sumoperators(operators, coefficients)

        dtype = canonicalize_dtypes(float, *operators, *coefficients, dtype=dtype)

        self._operators = tuple(operators)
        self._coefficients = jnp.asarray(coefficients, dtype=dtype)

        super().__init__(hi_spaces[0], self._coefficients.dtype)

        self._is_hermitian = all([op.is_hermitian for op in operators])
        self.__attrs = None

    @property
    def is_hermitian(self) -> bool:
        return self._is_hermitian

    @property
    def operators(self) -> tuple[ContinuousOperator, ...]:
        """The list of all operators in the terms of this sum. Every
        operator is summed with a corresponding coefficient
        """
        return self._operators

    @property
    def coefficients(self) -> Array:
        return self._coefficients

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

    def __repr__(self):
        return (
            f"SumOperator(operators={self.operators}, coefficients={self.coefficients})"
        )
