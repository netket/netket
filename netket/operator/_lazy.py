from ._abstract_operator import AbstractOperator


class WrappedOperator(AbstractOperator):
    def get_conn_flattened(self, *args):
        return NotImplementedError

    def __mul__(self, other):
        return self.collect() * other

    def __rmul__(self, other):
        return other * self.collect()

    def __matmul__(self, other):
        return self.collect() @ other

    def __rmatmul__(self, other):
        return other @ self.collect()

    def transpose(self):
        return self.T

    def conj(self):
        return self.conjugate()


class Transpose(WrappedOperator):
    """
    A wrapper lazily representing the transpose of the wrapped object.
    """

    parent: AbstractOperator
    """The wrapped object"""

    def __init__(self, op):
        super().__init__(op.hilbert)
        self.parent = op

    def collect(self):
        return self.parent._concrete_transpose()

    @property
    def T(self):
        return self.parent

    def conjugate(self):
        if self.parent.is_hermitian:
            return self.parent

        return Adjoint(self.parent)

    @property
    def H(self):
        return self.parent.conjugate()

    def _concrete_transpose(self):
        return self.parent

    def __repr__(self):
        return "Transpose({})".format(self.parent)

    def get_conn_flattened(self, *args):
        return NotImplementedError


class Adjoint(WrappedOperator):
    """
    A wrapper lazily representing the adjoint (conjugate-transpose) of the wrapped object.
    """

    parent: AbstractOperator
    """The wrapped object"""

    def __init__(self, op):
        super().__init__(op.hilbert)
        self.parent = op

    def collect(self):
        return self.parent._concrete_transpose().conj()

    @property
    def T(self):
        return self.parent.conjugate()

    def conjugate(self):
        return Transpose(self.parent)

    @property
    def H(self):
        return self.parent

    def _concrete_transpose(self):
        return self.parent.conj()

    def _concrete_adjoint(self):
        return self.parent._concrete_transpose.conj()

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

    def __matmul__(self, other):
        if self.parent == other:
            return Squared(other)

        return self.collect() @ other


class Squared(WrappedOperator):
    """
    A wrapper lazily representing the matrix-squared (A^dag A) of the wrapped object A.
    """

    parent: AbstractOperator
    """The wrapped object"""

    def __init__(self, op):
        super().__init__(op.hilbert)
        self.parent = op

    def collect(self):
        return self.parent.H._concrete_matmul_(self.parent)

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

    def _concrete_transpose(self):
        return self.parent.conj()

    def _concrete_adjoint(self):
        return self.parent._concrete_transpose.conj()

    def __repr__(self):
        return "Squared({})".format(self.parent)

    def __mul__(self, other):
        return self.collect() * other

    def __rmul__(self, other):
        return other * self.collect()

    def __matmul__(self, other):
        return self.collect() @ other
