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

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.utils.numbers import is_scalar
from netket.utils.types import DType, PyTree, Array

from netket.jax import canonicalize_dtypes
from netket.operator import ContinuousOperator
from netket.utils import HashableArray


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


@register_pytree_node_class
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

        self._is_hermitian = None
        self.__attrs = None

    @property
    def is_hermitian(self) -> bool:
        if self._is_hermitian is None:
            self._is_hermitian = all([op.is_hermitian for op in self.operators])
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

    def _expect_kernel(self, logpsi: Callable, params: PyTree, x: Array):
        result = [
            c * op._expect_kernel(logpsi, params, x)
            for c, op in zip(self.coefficients, self.operators)
        ]

        return sum(result)

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

    def tree_flatten(self):
        data = (self.operators, self.coefficients)
        metadata = {"dtype": self.dtype, "is_hermitian": self.is_hermitian}
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        (operators, coeffs) = data
        dtype = metadata["dtype"]
        is_hermitian = metadata["is_hermitian"]

        op = cls(*operators, coefficients=coeffs, dtype=dtype)
        op._is_hermitian = is_hermitian
        return op
