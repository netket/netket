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
from netket.utils import deprecated

from netket.hilbert import AbstractHilbert


class AbstractOperator(abc.ABC):
    """Abstract class for quantum Operators. This class prototypes the methods
    needed by a class satisfying the Operator concept.
    """

    _hilbert: AbstractHilbert
    r"""The hilbert space associated to this operator."""

    def __init__(self, hilbert: AbstractHilbert):
        self._hilbert = hilbert

    @property
    def hilbert(self) -> AbstractHilbert:
        r"""The hilbert space associated to this operator."""
        return self._hilbert

    # TODO: eventually remove this
    @property
    @deprecated(reason="Please use `operator.hilbert.size` instead.")
    def size(self) -> int:
        r"""The total number number of local degrees of freedom."""
        return self._hilbert.size

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
        Returns a guranteed concrete instancce of an operator.

        As some operations on operators return lazy wrapperes (such as transpose,
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
        return f"{type(self).__name__}(hilbert={self.hilbert})"
