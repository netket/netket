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

from typing import no_type_check
from collections.abc import Callable
from collections.abc import Hashable, Iterable

import jax.numpy as jnp

from netket.utils.numbers import is_scalar
from netket.utils.types import DType, PyTree, Array

from netket.jax import canonicalize_dtypes
from netket.operator import ContinuousOperator
from netket.utils import HashableArray, struct


def _flatten_sumoperators(
    operators: Iterable[ContinuousOperator], coefficients: Array
) -> tuple[list[ContinuousOperator], list[complex]]:
    """Flatten sumoperators inside of operators."""
    new_operators: list[ContinuousOperator] = []
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

    _operators: tuple[ContinuousOperator, ...]
    _coefficients: Array
    _is_hermitian: bool = struct.static_field()

    @no_type_check
    def __init__(
        self,
        *operators: tuple[ContinuousOperator, ...],
        coefficients: float | Iterable[float] = 1.0,
        dtype: DType | None = None,
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
            coefficients = [coefficients for _ in operators]  # type: ignore

        if len(operators) != len(coefficients):  # type: ignore
            raise AssertionError("Each operator needs a coefficient")

        operators, coefficients = _flatten_sumoperators(operators, coefficients)

        dtype = canonicalize_dtypes(float, *operators, *coefficients, dtype=dtype)

        self._operators = tuple(operators)  # type: tuple[ContinuousOperator, ...]
        self._coefficients = jnp.asarray(coefficients, dtype=dtype)
        self._is_hermitian = all([op.is_hermitian for op in operators])
        super().__init__(hi_spaces[0], self._coefficients.dtype)

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

    def _expect_kernel(
        self,
        logpsi: Callable,
        params: PyTree,
        x: Array,
    ) -> Array:
        result = [
            self.coefficients[i] * op._expect_kernel(logpsi, params, x)
            for i, op in enumerate(self.operators)
        ]
        return sum(result)

    @struct.property_cached(pytree_ignore=True)
    def _attrs(self) -> tuple[Hashable, ...]:
        return (  # type: ignore
            self.hilbert,
            self.operators,
            HashableArray(self.coefficients),
            self.dtype,
        )

    def __repr__(self):
        return (
            f"SumOperator(operators={self.operators}, coefficients={self.coefficients})"
        )
