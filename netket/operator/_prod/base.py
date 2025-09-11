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

from collections.abc import Iterable

from abc import ABC

import jax.numpy as jnp

from netket.hilbert import AbstractHilbert
from netket.utils.types import Array
from netket.jax import canonicalize_dtypes
from netket.utils.numbers import is_scalar

from netket.operator._abstract_operator import AbstractOperator
from netket.operator._discrete_operator import DiscreteOperator
from netket.operator._discrete_operator_jax import DiscreteJaxOperator

# from .._continuous_operator import ContinuousOperator


def _flatten_prodoperators(operators: Iterable[AbstractOperator], coefficient: float):
    """Flatten sumoperators inside of operators."""
    new_operators = []
    new_coeff = coefficient
    for op in operators:
        if isinstance(op, ProductOperator):
            new_operators.extend(op.operators)
            new_coeff = new_coeff * op.coefficient
        else:
            new_operators.append(op)
    return new_operators, new_coeff


class ProductOperator(ABC):
    def __new__(cls, *args, **kwargs):
        # This logic overrides the constructor, such that if someone tries to
        # construct this class directly by calling `SumOperator(...)`
        # it will construct either a DiscreteHilbert or TensorDiscreteHilbert
        from netket.operator._prod.operator import ProductGenericOperator
        from netket.operator._prod.discrete_operator import ProductDiscreteOperator
        from netket.operator._prod.discrete_jax_operator import ProductDiscreteJaxOperator

        # from .continuous import ProductContinuousOperator

        if cls is ProductOperator:
            if all(isinstance(op, DiscreteJaxOperator) for op in args):
                cls = ProductDiscreteJaxOperator
            elif all(isinstance(op, DiscreteOperator) for op in args):
                cls = ProductDiscreteOperator
            # elif all(isinstance(op, ContinuousOperator) for op in args):
            #    cls = ProductContinuousOperator
            else:
                cls = ProductGenericOperator
        return super().__new__(cls)

    def __init__(
        self,
        operators: Iterable[AbstractHilbert],
        *args,
        coefficient: float = 1.0,
        dtype=None,
        **kwargs,
    ):
        r"""Constructs a Sum of Operators.

        Args:
            *hilb: An iterable object containing at least 1 hilbert space.
        """
        hi_spaces = [op.hilbert for op in operators]
        if not all(hi == hi_spaces[0] for hi in hi_spaces):
            raise NotImplementedError(
                "Cannot construct a ProductOperator for operators on different Hilbert Spaces"
            )

        if not is_scalar(coefficient):
            raise TypeError(f"Coefficient must be a scalar but was {coefficient}")

        operators, coefficient = _flatten_prodoperators(operators, coefficient)

        dtype = canonicalize_dtypes(float, *operators, coefficient, dtype=dtype)

        self._operators = tuple(operators)
        self._coefficient = jnp.asarray(coefficient, dtype=dtype)
        self._dtype = dtype

        super().__init__(
            *args, **kwargs
        )  # forwards all unused arguments so that this class is a mixin.

    @property
    def dtype(self):
        return self._dtype

    @property
    def operators(self) -> tuple[AbstractOperator, ...]:
        """The tuple of all operators in the terms of this sum. Every
        operator is summed with a corresponding coefficient
        """
        return self._operators

    @property
    def coefficient(self) -> Array:
        return self._coefficient

    def __repr__(self) -> str:
        strs = [f"{type(self).__name__} with terms:"]
        # Format the coefficient
        coefficient_str = f" ∙ {self.coefficient}"
        strs.append(coefficient_str)

        # Format each operator
        for op in self.operators:
            op_str = str(op).splitlines()
            # Format the first line of the operator
            formatted_op = f" ∙ {op_str[0]}"
            # Add subsequent lines with proper indentation
            if len(op_str) > 1:
                formatted_op += "\n" + "\n".join(
                    f"{' ' * len(coefficient_str)}{line}" for line in op_str[1:]
                )
            strs.append(formatted_op)

        return "\n".join(strs)

    def __mul__(self, other):
        if is_scalar(other):
            return ProductOperator(
                *self.operators, coefficient=self.coefficient * other, dtype=self.dtype
            )
        return super().__mul__(other)

    def __matmul__(self, other):
        if not isinstance(other, AbstractOperator):
            return NotImplemented
            # return super().__matmul__(other)

        if isinstance(other, ProductOperator):
            ops = self.operators + other.operators
            coeff = self.coefficient * other.coefficient
            dtype = self.dtype if self.dtype == other.dtype else None
        else:
            ops = (*self.operators, other)
            coeff = self.coefficient
            dtype = self.dtype if self.dtype == other.dtype else None

        return ProductOperator(*ops, coefficient=coeff, dtype=dtype)

    def __rmatmul__(self, other):
        if not isinstance(other, AbstractOperator):
            return NotImplemented
            # return super().__matmul__(other)

        ops = (other, *self.operators)
        coeff = self.coefficient
        dtype = self.dtype if self.dtype == other.dtype else None

        return ProductOperator(*ops, coefficient=coeff, dtype=dtype)
