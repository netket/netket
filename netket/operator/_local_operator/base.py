# Copyright 2021-2022 The NetKet Authors - All rights reserved.
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

from typing import Union, Optional, TYPE_CHECKING

import numbers

from textwrap import dedent

import numpy as np
import jax.numpy as jnp
from scipy.sparse import issparse

from netket.hilbert import AbstractHilbert
from netket.utils.types import DType, Array
from netket.utils.numbers import dtype as _dtype, is_scalar

from .._discrete_operator import DiscreteOperator
from .._lazy import Transpose

from .helpers import (
    canonicalize_input,
    _multiply_operators,
    cast_operator_matrix_dtype,
)
from .convert import local_operators_to_pauli_strings

if TYPE_CHECKING:
    from .._pauli_strings import PauliStrings


def is_hermitian(a: np.ndarray, rtol=1e-05, atol=1e-08) -> bool:
    if issparse(a):
        return np.allclose(a.todense(), a.T.conj().todense(), rtol=rtol, atol=atol)
    else:
        return np.allclose(a, a.T.conj(), rtol=rtol, atol=atol)


def _is_sorted(a):
    for i in range(len(a) - 1):
        if a[i + 1] < a[i]:
            return False
    return True


class LocalOperatorBase(DiscreteOperator):
    """A custom local operator. This is a sum of an arbitrary number of operators
    acting locally on a limited set of k quantum numbers (i.e. k-local,
    in the quantum information sense).
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        operators: Union[list[Array], Array] = [],
        acting_on: Union[list[int], list[list[int]]] = [],
        constant: numbers.Number = 0,
        dtype: Optional[DType] = None,
    ):
        r"""
        Constructs a new ``LocalOperator`` given a hilbert space and (if
        specified) a constant level shift.

        Args:
           hilbert: Hilbert space the operator acts on.
           operators: A list of operators, in matrix form. Supports numpy dense or scipy
           sparse format
           acting_on: A list of list of sites, which the corresponding operators act on. This
                should be constructed such that :code:`operators[i]` acts on the sites :code:`acting_on[i]`.
                If operators is not a list of operators, acting_on should just be the list of
                corresponding sites.
           constant: Constant diagonal shift of the operator, equivalent to
                :math:`+\text{c}\hat{I}`. Default is 0.0.

        Examples:
           Constructs a ``LocalOperator`` without any operators.

           >>> from netket.hilbert import CustomHilbert
           >>> from netket.operator import LocalOperator
           >>> hi = CustomHilbert(local_states=[-1, 1])**20
           >>> empty_hat = LocalOperator(hi)
           >>> print(len(empty_hat.acting_on))
           0
        """
        super().__init__(hilbert)
        self.mel_cutoff = 1.0e-6
        self._initialized = None
        self._is_hermitian = None

        if not all(
            [_is_sorted(hilbert.states_at_index(i)) for i in range(hilbert.size)]
        ):
            raise ValueError(
                dedent(
                    """LocalOperator needs an hilbert space with sorted state values at
                every site.
                """
                )
            )

        # Canonicalize input. From now on input is guaranteed to be in canonical order
        operators, acting_on, dtype = canonicalize_input(
            self.hilbert, operators, acting_on, constant, dtype=dtype
        )
        self._dtype = dtype
        self._constant = np.array(constant, dtype=dtype)

        self._operators_dict = {}
        for op, aon in zip(operators, acting_on):
            self._add_operator(aon, op)

    def _add_operator(self, acting_on: tuple, operator: Array):
        """
        Adds an operator acting on a subset of sites.

        Does not modify in-place the operators themselves which are treated as
        immutables.
        """
        assert isinstance(acting_on, tuple)
        # acting_on_key = tuple(acting_on)
        if acting_on in self._operators_dict:
            operator = self._operators_dict[acting_on] + operator

        self._operators_dict[acting_on] = operator

    @property
    def operators(self) -> list[np.ndarray]:
        """List of the matrices of the operators encoded in this Local Operator.
        Returns a copy.
        """
        return list(self._operators_dict.values())

    @property
    def _operators(self) -> list[np.ndarray]:
        return self.operators

    @property
    def acting_on(self) -> list[list[int]]:
        """List containing the list of the sites on which every operator acts.

        Every operator `self.operators[i]` acts on the sites `self.acting_on[i]`
        """
        return list(self._operators_dict.keys())

    @property
    def n_operators(self) -> int:
        return len(self._operators_dict)

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_hermitian(self) -> bool:
        """Returns true if this operator is hermitian."""
        # TODO: (VolodyaCO) I guess that if we have an operator with diagonal elements equal to 1j*C+Y, some complex constant, and
        # self._constant=-1j*C, then the actual diagonal would be Y. How do we check hermiticity taking into account the diagonal
        # elements as well as the self._constant? For the moment I just check hermiticity of the added constant, which must be real.
        if self._is_hermitian is None:
            self._is_hermitian = all(map(is_hermitian, self.operators)) and np.isreal(
                self._constant
            )

        return self._is_hermitian

    @property
    def mel_cutoff(self) -> float:
        r"""float: The cutoff for matrix elements.
        Only matrix elements such that abs(O(i,i))>mel_cutoff
        are considered"""
        return self._mel_cutoff

    @mel_cutoff.setter
    def mel_cutoff(self, mel_cutoff):
        self._mel_cutoff = mel_cutoff
        assert self.mel_cutoff >= 0

    @property
    def constant(self) -> numbers.Number:
        return self._constant

    def to_pauli_strings(self) -> "PauliStrings":  # noqa: F821
        """Convert to PauliStrings object"""
        return local_operators_to_pauli_strings(
            self.hilbert, self.operators, self.acting_on, self.constant, self.dtype
        )

    def copy(self, *, dtype: Optional[DType] = None, _cls=None):
        """Returns a copy of the operator, while optionally changing the dtype
        of the operator.

        Args:
            dtype: optional dtype

        Internal args:
            _cls: used to specify the target class
        """

        if dtype is None:
            dtype = self.dtype

        if not np.can_cast(self.dtype, dtype, casting="same_kind"):
            raise ValueError(f"Cannot cast {self.dtype} to {dtype}")

        if _cls is None:
            _cls = type(self)

        new = _cls(self.hilbert, constant=self.constant, dtype=dtype)
        new.mel_cutoff = self.mel_cutoff

        if dtype == self.dtype:
            new._operators_dict = self._operators_dict.copy()
        else:
            new._operators_dict = {
                aon: cast_operator_matrix_dtype(op, dtype)
                for aon, op in self._operators_dict.items()
            }

        return new

    def transpose(self, *, concrete=False):
        r"""LocalOperator: Returns the transpose of this operator."""
        if concrete:
            new = self.copy()
            for aon in new._operators_dict.keys():
                new._operators_dict[aon] = new._operators_dict[aon].transpose()
            return new
        else:
            return Transpose(self)

    def conjugate(self, *, concrete=False):
        r"""LocalOperator: Returns the complex conjugate of this operator."""
        new = self.copy()
        for aon in new._operators_dict.keys():
            new._operators_dict[aon] = new._operators_dict[aon].copy().conjugate()
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __isub__(self, other):
        return self.__iadd__(-other)

    def __neg__(self):
        return -1 * self

    def __add__(self, other: Union["LocalOperatorBase", numbers.Number]):
        op = self.copy(dtype=jnp.promote_types(self.dtype, _dtype(other)))
        op = op.__iadd__(other)
        return op

    def __iadd__(self, other):
        if isinstance(other, LocalOperatorBase):
            if self.hilbert != other.hilbert:
                return NotImplemented

            if not np.can_cast(other.dtype, self.dtype, casting="same_kind"):
                raise ValueError(
                    f"Cannot add inplace operator with dtype {other.dtype} "
                    f"to operator with dtype {self.dtype}"
                )

            assert other.mel_cutoff == self.mel_cutoff
            self._constant += other.constant.item()
            for aon, op in other._operators_dict.items():
                self._add_operator(aon, op)

            self._reset_caches()
            return self
        if is_scalar(other):
            if not np.can_cast(type(other), self.dtype, casting="same_kind"):
                raise ValueError(
                    f"Cannot add inplace operator with dtype {type(other)} "
                    f"to operator with dtype {self.dtype}"
                )

            self._reset_caches()
            self._constant += other
            return self

        return NotImplemented

    def __truediv__(self, other):
        if not is_scalar(other):
            raise TypeError("Only division by a scalar number is supported.")

        if other == 0:
            raise ValueError("Dividing by 0")
        return self.__mul__(1.0 / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, DiscreteOperator):
            op = self.copy(dtype=jnp.promote_types(self.dtype, _dtype(other)))
            return op.__imatmul__(other)
        elif is_scalar(other):
            op = self.copy(dtype=jnp.promote_types(self.dtype, _dtype(other)))
            return op.__imul__(other)
        return NotImplemented

    def __imul__(self, other):
        if isinstance(other, DiscreteOperator):
            return self.__imatmul__(other)
        elif is_scalar(other):
            if not np.can_cast(_dtype(other), self.dtype, casting="same_kind"):
                raise ValueError(
                    f"Cannot add inplace operator of type {type(other)} and "
                    f"dtype {_dtype(other)} to operator with dtype {self.dtype}"
                )
            other = np.asarray(
                other, dtype=jnp.promote_types(self.dtype, _dtype(other))
            )

            self._constant *= other
            if np.abs(other) <= self.mel_cutoff:
                self._operators_dict = {}
            else:
                for key in self._operators_dict:
                    self._operators_dict[key] = other * self._operators_dict[key]

            self._reset_caches()
            return self

        return NotImplemented

    def __imatmul__(self, other):
        if not isinstance(other, LocalOperatorBase):
            return NotImplemented

        if not np.can_cast(other.dtype, self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot add inplace operator with dtype {type(other)} to operator with dtype {self.dtype}"
            )

        return self._op_imatmul_(other)

    def _op__matmul__(self, other: "LocalOperatorBase") -> "LocalOperatorBase":
        if not isinstance(other, LocalOperatorBase):
            return NotImplemented
        op = self.copy(dtype=jnp.promote_types(self.dtype, _dtype(other)))
        return op._op_imatmul_(other)

    def _op_imatmul_(self, other: "LocalOperatorBase") -> "LocalOperatorBase":
        if not isinstance(other, LocalOperatorBase):
            return NotImplemented

        # (α + ∑ᵢAᵢ)(β + ∑ᵢBᵢ) =
        # = αβ + α ∑ᵢBᵢ + β ∑ᵢAᵢ + ∑ᵢⱼAᵢBⱼ
        # = β(α + ∑ᵢAᵢ) + α ∑ᵢBᵢ + ∑ᵢⱼAᵢBⱼ

        α = self.constant.item()
        β = other.constant.item()
        # copy A dict because it is modified inplace in __imul__(β) and add_operators
        A_op_dict = self._operators_dict.copy()
        B_op_dict = other._operators_dict

        # αβ + β ∑ᵢAᵢ
        self.__imul__(β)

        # α ∑ᵢBᵢ
        if np.abs(α) > self.mel_cutoff:
            for aon, op in B_op_dict.items():
                self._add_operator(aon, α * op)

        # ∑ᵢⱼAᵢBⱼ
        for supp_A_i, A_i in A_op_dict.items():
            for supp_B_j, B_j in B_op_dict.items():
                self._add_operator(
                    *_multiply_operators(
                        self.hilbert, supp_A_i, A_i, supp_B_j, B_j, dtype=self.dtype
                    )
                )

        is_hermitian_A = self._is_hermitian
        is_hermitian_B = other._is_hermitian

        self._reset_caches()

        if is_hermitian_A and is_hermitian_B:
            self._is_hermitian = is_hermitian_A

        return self

    def _reset_caches(self):
        """
        Cleans the internal caches built on the operator.
        """
        self._initialized = False
        self._is_hermitian = None

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        self._setup()
        return self._max_conn_size

    def __repr__(self):
        ao = self.acting_on

        acting_str = f"acting_on={ao}"
        if len(acting_str) > 55:
            acting_str = f"#acting_on={len(ao)} locations"
        return f"{type(self).__name__}(dim={self.hilbert.size}, {acting_str}, constant={self.constant}, dtype={self.dtype})"
