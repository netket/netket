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
            # The |A_loc|² estimator requires the sampling distribution to cover
            # the support of A|psi>. A can leave a constrained sector even when
            # A†A preserves it, so form the actual product on constrained spaces.
            # (Adjoint can also wrap continuous operators, which lack `.constrained`.)
            is_constrained = (
                isinstance(self.hilbert, DiscreteHilbert) and self.hilbert.constrained
            )
            if not is_constrained:
                return Squared(other)
            try:
                adjoint = self.collect()
            except NotImplementedError as err:
                raise NotImplementedError(
                    f"Cannot compute A.H @ A on the constrained space {self.hilbert} "
                    f"because the adjoint of {type(other).__name__} cannot be "
                    "constructed explicitly.\n"
                    "The explicit `nk.operator.Squared(A)` estimator is an "
                    "alternative only if the support of A|psi> is contained "
                    "in the support of |psi>."
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
    r"""
    A wrapper lazily representing the matrix-squared (A^dag A) of the wrapped object A.

    Monte Carlo expectation values use :math:`|A_{loc}|^2`, which only requires
    the connected elements of A. This estimator is valid when the support of
    :math:`A|\psi\rangle` is contained in the support of :math:`|\psi\rangle`.
    A constraint-preserving A and a wavefunction with full support within that
    space satisfy this condition. Constraint preservation alone is not
    sufficient: exact zeros can invalidate the estimator within the sector.

    Dense and sparse conversion instead multiplies the matrix of A restricted
    to its Hilbert space. This requires A to map that space into itself, but
    has no wavefunction-support requirement. For a parent that changes the
    constraint, form the product with :meth:`collect` before conversion if
    concrete multiplication is supported.

    On constrained discrete spaces, :code:`A.H @ A` forms the actual product
    instead of automatically selecting this estimator. Explicit construction
    of :code:`Squared(A)` keeps the estimator described above.
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
