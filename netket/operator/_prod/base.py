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
    """
    Base class for product of quantum operators.

    This class represents a product of quantum operators with a coefficient,
    implementing the mathematical concept of :math:`c \\prod_i \\hat{H}_i` where
    :math:`c` is a scalar coefficient and :math:`\\hat{H}_i` are quantum operators.

    The class uses a dispatch mechanism to automatically select the appropriate
    specialized subclass based on the types of operators being multiplied:

    * :class:`~netket.operator._prod.discrete_jax_operator.ProductDiscreteJaxOperator`
      for products of :class:`~netket.operator.DiscreteJaxOperator` instances
    * :class:`~netket.operator._prod.discrete_operator.ProductDiscreteOperator`
      for products of :class:`~netket.operator.DiscreteOperator` instances
    * :class:`~netket.operator._prod.operator.ProductGenericOperator`
      for mixed operator types or fallback cases.

    Products can be constructed using the ``@`` operator (matrix multiplication) or by
    directly instantiating this class with a list of operators.

    Examples:
        Creating a product of Pauli operators:

        >>> import netket as nk
        >>> hi = nk.hilbert.Spin(s=1/2, N=4)
        >>> sx0 = nk.operator.spin.sigmax(hi, 0)
        >>> sz1 = nk.operator.spin.sigmaz(hi, 1)
        >>> sy2 = nk.operator.spin.sigmay(hi, 2)
        >>> # Using ProductOperator constructor
        >>> product1 = nk.operator.ProductOperator(sx0, sz1, sy2)
        >>> print(product1)  # doctest: +SKIP
        ProductDiscreteJaxOperator with terms:
         ∙ 1.0
         ∙ PauliStringsJax(n_sites=4, hilbert=Spin(s=1/2, N=4), n_strings=1)
         ∙ PauliStringsJax(n_sites=4, hilbert=Spin(s=1/2, N=4), n_strings=1)
         ∙ PauliStringsJax(n_sites=4, hilbert=Spin(s=1/2, N=4), n_strings=1)

        Creating a product with explicit coefficient:

        >>> # Direct instantiation
        >>> product2 = nk.operator.ProductOperator(sx0, sz1, coefficient=2.0)
        >>> print(product2.coefficient)
        2.0

        Products of products are automatically flattened:

        >>> prod_a = nk.operator.ProductOperator(sx0, sz1)
        >>> prod_b = nk.operator.ProductOperator(sy2, nk.operator.spin.sigmaz(hi, 3))
        >>> combined = nk.operator.ProductOperator(prod_a, prod_b)
        >>> print(len(combined.operators))  # All 4 operators are flattened
        4

        Scaling a product:

        >>> scaled = 3.0 * product1
        >>> print(scaled.coefficient)
        (3+0j)
    """

    def __new__(cls, *args, **kwargs):
        # This logic overrides the constructor, such that if someone tries to
        # construct this class directly by calling `SumOperator(...)`
        # it will construct either a DiscreteHilbert or TensorDiscreteHilbert
        from netket.operator._prod.operator import ProductGenericOperator
        from netket.operator._prod.discrete_operator import ProductDiscreteOperator
        from netket.operator._prod.discrete_jax_operator import (
            ProductDiscreteJaxOperator,
        )

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
        r"""Constructs a Product of Operators.

        Args:
            operators: An iterable of quantum operators to be multiplied. All operators
                must act on the same Hilbert space.
            coefficient: Scalar coefficient for the product. Default is 1.0.
            dtype: Data type for the coefficient. If None, it will be inferred
                from the operators and coefficient.
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
        """The tuple of all operators in the terms of this product. All operators
        are multiplied together with the scalar coefficient.
        """
        return self._operators

    @property
    def coefficient(self) -> Array:
        """The scalar coefficient multiplying the product of operators."""
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
            return super().__matmul__(other)

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
            return super().__rmatmul__(other)

        ops = (other, *self.operators)
        coeff = self.coefficient
        dtype = self.dtype if self.dtype == other.dtype else None

        return ProductOperator(*ops, coefficient=coeff, dtype=dtype)
