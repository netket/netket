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

import abc

from netket.utils.types import DType

from netket.hilbert import AbstractHilbert

from ._abstract_observable import AbstractObservable


class AbstractOperator(AbstractObservable):
    """Abstract class for quantum Operators.

    An operator is a general object that defines a linear transformation
    on vectors of the Hilbert space and their expectation value can
    be comptued starting from a variational state using the method
    `expect` or `expect_and_grad` of the variational states.

    If the object is not a linear operator but some more general
    function, it should inherit from {class}`~netket.operator.AbstractObservable`
    instead.

    Operators over discrete hilbert spaces that can be converted
    to a dense representation should instead inherit from
    {class}`~netket.operator.DiscreteOperator`, while operators
    over that only work over continuous hilbert spaces should
    inherit from {class}`~netket.operator.ContinuousOperator`.

    This class determines the basic methods that an operator
    must implement to work correctly with NetKet.
    """

    def __init__(self, hilbert: AbstractHilbert):
        super().__init__(hilbert)

    @property
    def is_hermitian(self) -> bool:
        """Returns true if this operator is hermitian."""
        return False

    @property
    def H(self) -> "AbstractOperator":
        """Returns the Conjugate-Transposed operator"""
        if self.is_hermitian:
            return self

        from ._lazy import Adjoint

        return Adjoint(self)

    @property
    def T(self) -> "AbstractOperator":
        """Returns the transposed operator"""
        return self.transpose()

    @property
    @abc.abstractmethod
    def dtype(self) -> DType:
        """The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩."""

    def collect(self) -> "AbstractOperator":
        """
        Returns a guaranteed concrete instance of an operator.

        As some operations on operators return lazy wrappers (such as transpose,
        hermitian conjugate...), this is used to obtain a guaranteed non-lazy
        operator.
        """
        return self

    def transpose(self, *, concrete=False) -> "AbstractOperator":
        """Returns the transpose of this operator.

        Args:
            concrete: if True returns a concrete operator and not a lazy wrapper

        Returns:
            if concrete is not True, self or a lazy wrapper; the
            transposed operator otherwise
        """
        if not concrete:
            from ._lazy import Transpose

            return Transpose(self)
        else:
            raise NotImplementedError

    def conjugate(self, *, concrete=False) -> "AbstractOperator":
        """Returns the complex-conjugate of this operator.

        Args:
            concrete: if True returns a concrete operator and not a lazy wrapper

        Returns:
            if concrete is not True, self or a lazy wrapper; the
            complex-conjugated operator otherwise
        """
        raise NotImplementedError

    def conj(self, *, concrete=False) -> "AbstractOperator":
        return self.conjugate(concrete=False)

    def __repr__(self):
        return f"{type(self).__name__}(hilbert={self.hilbert}, dtype={self.dtype})"
