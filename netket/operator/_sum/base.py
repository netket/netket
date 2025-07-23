# Copyright 2025 The NetKet Authors - All rights reserved.
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

from typing import Union
from collections.abc import Iterable

from abc import ABCMeta

import jax.numpy as jnp
from jax.stages import ArgInfo

from netket.hilbert import AbstractHilbert
from netket.utils.types import Array
from netket.jax import canonicalize_dtypes
from netket.utils.numbers import is_scalar

from netket.operator._abstract_operator import AbstractOperator
from netket.operator._discrete_operator import DiscreteOperator
from netket.operator._discrete_operator_jax import DiscreteJaxOperator
from netket.operator._continuous_operator import ContinuousOperator


def _flatten_sumoperators(operators: Iterable[AbstractOperator], coefficients: Array):
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


class SumOperatorMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        __tracebackhide__ = True
        # This logic overrides the constructor, such that if someone tries to
        # construct this class directly by calling `SumOperator(...)`
        # it will construct either a DiscreteHilbert or TensorDiscreteHilbert
        from .operator import SumGenericOperator
        from .discrete_operator import SumDiscreteOperator
        from .discrete_jax_operator import SumDiscreteJaxOperator
        from .continuous import SumContinuousOperator

        if cls is SumOperator:
            if all(isinstance(op, DiscreteJaxOperator) for op in args):
                cls = SumDiscreteJaxOperator
            elif all(isinstance(op, DiscreteOperator) for op in args):
                cls = SumDiscreteOperator
            elif all(isinstance(op, ContinuousOperator) for op in args):
                cls = SumContinuousOperator
            else:
                cls = SumGenericOperator
            return cls(*args, **kwargs)
        else:
            # If this is the Metaclass of a subclass, dispatch directly
            return super().__call__(*args, **kwargs)


class SumOperator(metaclass=SumOperatorMeta):
    """
    Base class for sum of quantum operators.

    This class represents a linear combination of quantum operators with coefficients,
    implementing the mathematical concept of :math:`\\sum_i c_i \\hat{H}_i` where
    :math:`c_i` are scalar coefficients and :math:`\\hat{H}_i` are quantum operators.

    The class uses a metaclass dispatch mechanism to automatically select the
    appropriate specialized subclass based on the types of operators being summed:

    * :class:`~netket.operator._sum.discrete_jax_operator.SumDiscreteJaxOperator`
      for sums of :class:`~netket.operator.DiscreteJaxOperator` instances
    * :class:`~netket.operator._sum.discrete_operator.SumDiscreteOperator`
      for sums of :class:`~netket.operator.DiscreteOperator` instances
    * :class:`~netket.operator._sum.continuous.SumContinuousOperator`
      for sums of :class:`~netket.operator.ContinuousOperator` instances
    * :class:`~netket.operator._sum.operator.SumGenericOperator`
      for mixed operator types or fallback cases.
    """

    _operators: tuple[ContinuousOperator, ...]
    _coefficients: Array
    _is_hermitian: bool

    def __init__(
        self,
        operators: Iterable[AbstractHilbert],
        *args,
        coefficients: Union[float, Iterable[float]] = 1.0,
        dtype=None,
        **kwargs,
    ):
        r"""Constructs a Sum of Operators.

        Args:
            *operators: An iterable of quantum operators to be summed. All operators
                must act on the same Hilbert space.
            coefficients: Scalar coefficient or iterable of coefficients for each
                operator. If a single scalar is provided, it will be used for all
                operators. Default is 1.0.
            dtype: Data type for the coefficients. If None, it will be inferred
                from the operators and coefficients.
        """
        hi_spaces = [op.hilbert for op in operators]
        if not all(hi == hi_spaces[0] for hi in hi_spaces):
            raise NotImplementedError(
                "Cannot construct a SumOperator for operators on different Hilbert Spaces"
            )

        if is_scalar(coefficients):
            coefficients = [coefficients for _ in operators]

        # ArgInfo shows up in packing with .lower() sometimes... it breaks all
        if not isinstance(coefficients, ArgInfo):
            if len(operators) != len(coefficients):
                raise AssertionError("Each operator needs a coefficient")

            operators, coefficients = _flatten_sumoperators(operators, coefficients)

            dtype = canonicalize_dtypes(*operators, *coefficients, dtype=dtype)
            coefficients = jnp.asarray(coefficients, dtype=dtype)

        self._operators = tuple(operators)
        self._coefficients = coefficients

        # Call parent classes without dtype parameter - we handle dtype ourselves
        super().__init__(
            *args, **kwargs
        )  # forwards all unused arguments so that this class is a mixin.

        # Set our computed dtype after parent __init__ methods have run
        # This ensures our dtype takes precedence over any parent class dtype computation
        self._dtype = dtype

    @property
    def dtype(self):
        """The data type of the operator coefficients.

        Returns:
            The numpy/JAX dtype used for the coefficients array.
        """
        return self._dtype

    @property
    def operators(self) -> tuple[AbstractOperator, ...]:
        r"""The tuple of all operators in the terms of this sum.

        Each operator corresponds to a term :math:`\hat{O}_i` in the sum
        :math:`\sum_i c_i \hat{O}_i`, where each operator is multiplied by
        its corresponding coefficient from :attr:`coefficients`.

        Returns:
            A tuple containing all the quantum operators being summed.
        """
        return self._operators

    @property
    def coefficients(self) -> Array:
        r"""The coefficients for each operator term in the sum.

        Each coefficient corresponds to :math:`c_i` in the sum
        :math:`\sum_i c_i \hat{O}_i`, where each coefficient multiplies
        its corresponding operator from :attr:`operators`.

        Returns:
            A JAX array containing the scalar coefficients for each operator.
        """
        return self._coefficients

    def __repr__(self) -> str:
        strs = [f"{type(self).__name__} with terms:"]
        for op, c in zip(self.operators, self.coefficients):
            op_str = str(op).splitlines()
            # Format the first line with the coefficient and operator
            formatted_op = f" ∙ {c} * {op_str[0]}"
            # Add subsequent lines with proper indentation
            if len(op_str) > 1:
                formatted_op += "\n" + "\n".join(
                    f"   {' ' * len(str(c))}   {line}" for line in op_str[1:]
                )
            strs.append(formatted_op)
        return "\n".join(strs)

    def __add__(self, other):
        if not isinstance(other, AbstractOperator):
            return NotImplemented

        if isinstance(other, SumOperator):
            ops = self.operators + other.operators
            coeffs = jnp.concatenate([self.coefficients, other.coefficients])
            dtype = self.dtype if self.dtype == other.dtype else None
        else:
            ops = (*self.operators, other)
            coeffs = jnp.concatenate([self.coefficients, jnp.array([1.0])])
            dtype = self.dtype if self.dtype == other.dtype else None

        return SumOperator(*ops, coefficients=coeffs, dtype=dtype)
