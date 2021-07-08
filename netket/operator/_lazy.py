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

from netket.utils.dispatch import parametric

from ._abstract_operator import AbstractOperator


class WrappedOperator(AbstractOperator):
    def get_conn_flattened(self, *args):
        return NotImplementedError

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

    def _op__matmul__(self, other):
        return self.collect() @ other

    def _op__rmatmul__(self, other):
        return other @ self.collect()


@parametric
class Transpose(WrappedOperator):
    """
    A wrapper lazily representing the transpose of the wrapped object.
    """

    parent: AbstractOperator
    """The wrapped object"""

    def __init__(self, op):
        super().__init__(op.hilbert)
        self.parent = op

    @property
    def dtype(self):
        return self.parent.dtype

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

    def __repr__(self):
        return "Transpose({})".format(self.parent)

    def get_conn_flattened(self, *args):
        return NotImplementedError

    def to_sparse(self):
        return self.parent.to_sparse().T


@parametric
class Adjoint(WrappedOperator):
    """
    A wrapper lazily representing the adjoint (conjugate-transpose) of the wrapped object.
    """

    parent: AbstractOperator
    """The wrapped object"""

    def __init__(self, op):
        super().__init__(op.hilbert)
        self.parent = op

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
        return "Adjoint({})".format(self.parent)

    def __mul__(self, other):
        if self.parent == other:
            return Squared(other)

        return self.collect() * other

    def __rmul__(self, other):
        if self.parent == other:
            return Squared(other.H.collect())

        return other * self.collect()

    def _op__matmul__(self, other):
        if self.parent == other:
            return Squared(other)

        return self.collect() @ other

    # def __op_rmatmul__(self, other):
    #    if self.parent == other:
    #        return Squared(other.H.collect())
    #
    #    return other @ self.collect()

    def to_sparse(self):
        return self.parent.to_sparse().T.conj()


@parametric
class Squared(WrappedOperator):
    """
    A wrapper lazily representing the matrix-squared (A^dag A) of the wrapped object A.
    """

    parent: AbstractOperator
    """The wrapped object"""

    def __init__(self, op):
        super().__init__(op.hilbert)
        self.parent = op

    @property
    def dtype(self):
        return self.parent.dtype

    def collect(self):
        return self.parent.H.collect()._concrete_matmul_(self.parent)

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
        return "Squared({})".format(self.parent)

    def __mul__(self, other):
        return self.collect() * other

    def __rmul__(self, other):
        return other * self.collect()

    def _op__matmul__(self, other):
        return self.collect() @ other

    def to_sparse(self):
        op = self.parent.to_sparse()
        return op.conj().T @ op
