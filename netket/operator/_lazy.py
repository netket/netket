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

import numpy as np
import jax.numpy as jnp

from netket.hilbert import DiscreteHilbert
from netket.utils.dispatch import parametric

from ._abstract_operator import AbstractOperator
from ._discrete_operator import DiscreteOperator


def _is_constrained(hilbert) -> bool:
    # Adjoint can also wrap continuous operators, whose spaces lack `.constrained`.
    return isinstance(hilbert, DiscreteHilbert) and hilbert.constrained


class WrappedOperator(AbstractOperator):
    _parent: DiscreteOperator
    """The wrapped object"""

    def __init__(self, op):
        super().__init__(op.hilbert)
        self._parent = op

    @property
    def parent(self) -> AbstractOperator:
        """The wrapped operator"""
        return self._parent

    @property
    def dtype(self):
        return self.parent.dtype

    def __add__(self, other):
        return self.collect() + other

    def __radd__(self, other):
        return other + self.collect()

    def __sub__(self, other):
        return self.collect() - other

    def __rsub__(self, other):
        return other - self.collect()

    def __mul__(self, other):
        return self.collect() * other

    def __rmul__(self, other):
        return other * self.collect()

    def __matmul__(self, other):
        # this is a base implementation of matmul
        if isinstance(other, AbstractOperator):
            if self == other and self.is_hermitian:
                from ._lazy import Squared

                return Squared(self)
            else:
                return self._op__matmul__(other)
        if isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray):
            return self._op__matmul__(other)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, AbstractOperator):
            if self == other and self.is_hermitian:
                from ._lazy import Squared

                return Squared(self)
            else:
                return self._op__rmatmul__(other)
        else:
            return NotImplemented

    def _op__matmul__(self, other):
        return self.collect() @ other

    def _op__rmatmul__(self, other):
        return other @ self.collect()

    def to_sparse(self):
        if not hasattr(self.parent, "to_sparse"):
            raise TypeError(
                "The Transposed Operator cannot be converted to a sparse Matrix"
            )
        return self._process_parent_array(self.parent.to_sparse())

    def to_dense(self):
        if not hasattr(self.parent, "to_dense"):
            raise TypeError(
                "The Transposed Operator cannot be converted to a dense Matrix"
            )
        return self._process_parent_array(self.parent.to_dense())

    def __repr__(self):
        return f"{type(self).__name__}({self.parent})"


@parametric
class Transpose(WrappedOperator):
    """
    A wrapper lazily representing the transpose of the wrapped object.
    """

    def collect(self):
        return self.parent.transpose(concrete=True)

    def transpose(self, *, concrete=False):
        return self.parent

    def conjugate(self, *, concrete=False):
        if self.parent.is_hermitian:
            return self.parent

        if concrete:
            return self.parent.conjugate(concrete=True)

        return Adjoint(self.parent)

    @property
    def H(self):
        return self.parent.conjugate()

    def _process_parent_array(self, op):
        """transposes the dense/sparse parent array"""
        return op.T


@parametric
class Adjoint(WrappedOperator):
    """
    A wrapper lazily representing the adjoint (conjugate-transpose) of the wrapped object.
    """

    parent: DiscreteOperator
    """The wrapped object"""

    @property
    def dtype(self):
        return self.parent.dtype

    def collect(self):
        return self.parent.transpose(concrete=True).conj(concrete=True)

    def transpose(self, *, concrete=False):
        return self.parent.conjugate(concrete=concrete)

    def conjugate(self, concrete=False):
        return self.parent.transpose(concrete=concrete)

    @property
    def H(self):
        return self.parent

    def __repr__(self):
        return f"Adjoint({self.parent})"

    def __mul__(self, other):
        return self.collect() * other

    def __rmul__(self, other):
        return other * self.collect()

    def _op__matmul__(self, other):
        if self.parent == other:
            # Squared(A) estimates A†A from |A_loc|², which is only correct if A
            # preserves the constraint. As we cannot check that, on constrained
            # spaces we form the actual product instead.
            if not _is_constrained(self.hilbert):
                return Squared(other)
            try:
                adjoint = self.collect()
            except NotImplementedError as err:
                raise NotImplementedError(
                    f"Cannot compute A.H @ A on the constrained space {self.hilbert} "
                    f"because the adjoint of {type(other).__name__} cannot be "
                    "constructed explicitly.\n"
                    "If A preserves the constraint, use `nk.operator.Squared(A)`, "
                    "which estimates A†A from |A_loc|²."
                ) from err
            return adjoint @ other

        return self.collect() @ other

    # def __op_rmatmul__(self, other):
    #    if self.parent == other:
    #        return Squared(other.H.collect())
    #
    #    return other @ self.collect()

    def _process_parent_array(self, op):
        """computes the adjoint of the dense/sparse parent array"""
        return op.T.conj()


@parametric
class Squared(WrappedOperator):
    """
    A wrapper lazily representing the matrix-squared (A^dag A) of the wrapped object A.

    Expectation values are estimated from :math:`|A_{loc}|^2`, which only
    requires the connected elements of A. On constrained Hilbert spaces this
    (as well as the conversion to a dense or sparse matrix) is only correct if
    A preserves the constraint. For this reason, :code:`A.H @ A` returns the
    actual product rather than :code:`Squared(A)` on constrained spaces.
    """

    parent: DiscreteOperator
    """The wrapped object"""

    @property
    def dtype(self):
        return self.parent.dtype

    def collect(self):
        return self.parent.H.collect()._op__matmul__(self.parent)

    @property
    def T(self):
        self.parent.c @ self.parent.T
        self.parent
        return self.parent.conjugate()

    def conjugate(self):
        return Transpose(self.parent)

    @property
    def H(self):
        return self

    @property
    def is_hermitian(self) -> bool:
        return True

    def __repr__(self):
        return f"Squared({self.parent})"

    def __mul__(self, other):
        return self.collect() * other

    def __rmul__(self, other):
        return other * self.collect()

    def _op__matmul__(self, other):
        return self.collect() @ other

    def _process_parent_array(self, op):
        """computes the matrix square of the dense/sparse parent array"""
        return op.conj().T @ op
