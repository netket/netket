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

import re
from typing import Iterable, List, Union, Optional
from netket.utils.types import DType, Array

import numpy as np
from numba import jit
from itertools import product

from netket.hilbert import Qubit, AbstractHilbert
from netket.utils.numbers import is_scalar

from .._abstract_operator import AbstractOperator
from .._discrete_operator import DiscreteOperator

valid_pauli_regex = re.compile(r"^[XYZI]+$")


def _standardize_matrix_input_type(op):
    """
    Standardize the structural type of operators stored in LocalOperator.

    Eventually, we could also support spmatrices (but some work will be needed.)
    """
    if isinstance(op, list):
        return np.asarray(op)
    else:
        return op


def cast_operator_matrix_dtype(matrix: Array, dtype: DType):
    """
    Changes the dtype of a matrix, without changing the structural type of the object.

    This makes sure that if you pass sparse arrays to a LocalOperator, they remain
    sparse even if you change the dtype
    """
    # must copy
    # return np.asarray(matrix, dtype=dtype)
    return matrix.astype(dtype)


def canonicalize_input(hilbert: AbstractHilbert, operators, weights, *, dtype=None):
    if operators is None:
        raise ValueError(
            "None valued operators passed. (Might arise when passing None valued hilbert explicitly)"
        )

    # Support single-operator
    if isinstance(operators, str):
        operators = [operators]

    if len(operators) == 0:
        raise ValueError("No Pauli operators passed.")

    # default weight is 1
    if weights is None:
        weights = True

    if is_scalar(weights):
        weights = [weights for _ in operators]

    if len(weights) != len(operators):
        raise ValueError("weights should have the same length as operators.")

    _hilb_size = len(operators[0])
    consistent = all(len(op) == _hilb_size for op in operators)
    if not consistent:
        raise ValueError("Pauli strings have inhomogeneous lengths.")

    consistent = all(bool(valid_pauli_regex.search(op)) for op in operators)
    if not consistent:
        raise ValueError(
            """Operators in string must be one of
            the Pauli operators X,Y,Z, or the identity I"""
        )

    if hilbert is None:
        hilbert = Qubit(_hilb_size)

    if not np.allclose(hilbert.shape, 2):
        raise ValueError(
            "PauliStrings only work for local hilbert size 2 where PauliMatrices are defined"
        )

    weights = _standardize_matrix_input_type(weights)
    weights = cast_operator_matrix_dtype(weights, dtype=dtype)

    operators = np.asarray(operators, dtype=str)

    return hilbert, operators, weights, weights.dtype


class PauliStringsBase(DiscreteOperator):
    """A Hamiltonian consisting of the sum of products of Pauli operators."""

    def __init__(
        self,
        hilbert: AbstractHilbert,
        operators: Union[str, List[str]] = None,
        weights: Union[float, complex, List[Union[float, complex]]] = None,
        *,
        cutoff: float = 1.0e-10,
        dtype: Optional[DType] = None,
    ):
        """
        Constructs a new ``PauliStrings`` operator given a set of Pauli operators.
        This class has two possible forms for initialization: ``PauliStrings(hilbert, operators, ...)`` or  ``PauliStrings(operators, ...)``.
        When no hilbert argument is passed, the hilbert defaults to Qubit, where the number of qubits is automatically deduced from the operators.

        Args:
           hilbert: A hilbert space, optional (is no ``AbstractHilbert`` is passed, default is Qubit)
           operators (list(string)): A list of Pauli operators in string format, e.g. ['IXX', 'XZI'].
           weights: A list of amplitudes of the corresponding Pauli operator.
           cutoff (float): a cutoff to remove small matrix elements

        Examples:
           Constructs a new ``PauliStrings`` operator X_0*X_1 + 3.*Z_0*Z_1 with both construction schemes.

           >>> import netket as nk
           >>> operators, weights = ['XX','ZZ'], [1,3]
           >>> op = nk.operator.PauliStrings(operators, weights)
           >>> op.hilbert
           Qubit(N=2)
           >>> op.hilbert.size
           2
           >>> hilbert = nk.hilbert.Spin(1/2, 2)
           >>> op = nk.operator.PauliStrings(hilbert, operators, weights)
           >>> op.hilbert
           Spin(s=1/2, N=2)
        """
        if hilbert is None:
            raise ValueError("None-valued hilbert passed.")

        # if first argument is not Hilbert, then shift all arguments by one
        # to support not declaring the Hilbert space
        if not isinstance(hilbert, AbstractHilbert):
            hilbert, operators, weights = None, hilbert, operators

        hilbert, operators, weights, dtype = canonicalize_input(
            hilbert, operators, weights, dtype=dtype
        )

        if not np.isscalar(cutoff) or cutoff < 0:
            raise ValueError("invalid cutoff in PauliStrings.")

        super().__init__(hilbert)

        self._operators = operators
        self._weights = weights
        self._dtype = dtype

        self._cutoff = cutoff

        self._is_hermitian = None

    @property
    def operators(self) -> Iterable[str]:
        return self._operators

    @property
    def weights(self) -> Iterable[str]:
        return self._weights

    @classmethod
    def identity(cls, hilbert: AbstractHilbert, **kwargs):
        return cls(hilbert, "I" * hilbert.size, **kwargs)

    @classmethod
    def from_openfermion(
        cls,
        hilbert: AbstractHilbert,
        of_qubit_operator=None,  # : "openfermion.ops.QubitOperator" type
        *,
        n_qubits: int = None,
    ) -> "PauliStringsBase":
        r"""
        Converts an openfermion QubitOperator into a netket PauliStrings.

        The hilbert first argument can be dropped, see :code:`__init__` for
        details and default value

        Args:
            hilbert: hilbert of the resulting PauliStrings object
            of_qubit_operator: this must be a
                `QubitOperator object <https://quantumai.google/reference/python/openfermion/ops/QubitOperator>`_ .
                More information about those objects can be found in
                `OpenFermion's documentation <https://quantumai.google/reference/python/openfermion>`_
            n_qubits: (optional) total number of qubits in the system, default None means inferring
                it from the QubitOperator. Argument is ignored when hilbert is given.

        """
        from openfermion.ops import QubitOperator

        if hilbert is None:
            raise ValueError("None-valued hilbert passed.")

        if not isinstance(hilbert, AbstractHilbert):
            # if first argument is not Hilbert, then shift all arguments by one
            hilbert, of_qubit_operator = None, hilbert

        if not isinstance(of_qubit_operator, QubitOperator):
            raise NotImplementedError()
        operators = []
        weights = []
        if hilbert is not None:
            # no warning, just overwrite
            n_qubits = hilbert.size
        if n_qubits is None:
            n_qubits = _count_of_locations(of_qubit_operator)
        for operator, weight in of_qubit_operator.terms.items():  # gives dict
            s = ["I"] * n_qubits
            for loc, op in operator:
                assert (
                    loc < n_qubits
                ), f"operator index {loc} is longer than n_qubits={n_qubits}"
                s[loc] = op
            operators.append("".join(s))
            weights.append(weight)

        ps_args = (operators, weights)
        if hilbert is not None:
            ps_args = (hilbert,) + ps_args
        return cls(*ps_args)

    @property
    def dtype(self) -> DType:
        """The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩."""
        return self._dtype

    @property
    def is_hermitian(self) -> bool:
        """Returns true if this operator is hermitian."""
        if self._is_hermitian is None:
            self._is_hermitian = np.allclose(self._weights.imag, 0.0)
        return self._is_hermitian

    def __repr__(self):
        print_list = []
        for op, w in zip(self._operators, self._weights):
            print_list.append(f"    {op} : {str(w)}")
        s = "{}(hilbert={}, n_strings={}, dict(operators:weights)=\n{}\n)".format(
            type(self).__name__,
            self.hilbert,
            len(self._operators),
            ",\n".join(print_list),
        )
        return s

    def copy(self, *, dtype: Optional[DType] = None):
        """Returns a copy of the operator, while optionally changing the dtype
        of the operator.

        Args:
            dtype: optional dtype
        """

        if dtype is None:
            dtype = self.dtype

        if not np.can_cast(self.dtype, dtype, casting="same_kind"):
            raise ValueError(f"Cannot cast {self.dtype} to {dtype}")

        new = type(self)(self.hilbert, dtype=dtype)
        new.mel_cutoff = self.mel_cutoff
        new._operators = self._operators

        if dtype == self.dtype:
            new._weights = self._weights.copy()
        else:
            new._weights = cast_operator_matrix_dtype(self._weights, dtype)

        return new

    def _op__matmul__(self, other):
        if not isinstance(other, PauliStringsBase):
            return NotImplementedError
        if not self.hilbert == other.hilbert:
            raise ValueError(
                f"Can only multiply identical hilbert spaces (got A@B, A={self.hilbert}, B={other.hilbert})"
            )
        operators, weights = _matmul(
            self._operators,
            self._weights,
            other._operators,
            other._weights,
        )
        return type(self)(
            self.hilbert,
            operators,
            weights,
            cutoff=max(self._cutoff, other._cutoff),
            dtype=self.dtype,
        )

    def __rmul__(self, scalar):
        return self * scalar

    def __mul__(self, scalar):
        if isinstance(scalar, AbstractOperator):
            raise TypeError(
                "To multiply operators use the matrix`@` "
                "multiplication operator `@` instead of the element-wise "
                "multiplication operator `*`.\n\n"
                "For example:\n\n"
                ">>> nk.operator.PauliStrings('XY')@nk.operator.PauliStrings('ZY')"
                "\n\n"
            )

        if not np.issubdtype(type(scalar), np.number):
            raise NotImplementedError
        weights = self._weights * scalar
        return type(self)(
            self.hilbert,
            self._operators,
            weights,
            dtype=self.dtype,
            cutoff=self._cutoff,
        )

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return -1 * self

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        if np.issubdtype(type(other), np.number):
            if other != 0.0:
                # adding a constant = adding IIII...III with weight being the constant
                return self + self.identity(self.hilbert) * other
            return self
        if not isinstance(other, PauliStringsBase):
            raise NotImplementedError
        if not self.hilbert == other.hilbert:
            raise ValueError(
                f"Can only add identical hilbert spaces (got A+B, A={self.hilbert}, B={other.hilbert})"
            )
        operators = np.concatenate((self._operators, other._operators))
        weights = np.concatenate((self._weights, other._weights))
        operators, weights = _reduce_pauli_string(operators, weights)
        return type(self)(
            self.hilbert,
            operators,
            weights,
            dtype=self.dtype,
            cutoff=self._cutoff,
        )


def _count_of_locations(of_qubit_operator):
    """Obtain the number of qubits in the openfermion QubitOperator. Openfermion builds operators from terms that store operators locations.
    Args:
        of_qubit_operator (openfermion.QubitOperator, openfermion.FermionOperator)
    Returns:
        n_qubits (int): number of qubits in the operator, which we can use to create a suitable hilbert space
    """
    # we always start counting from 0, so we only determine the maximum location
    def max_or_default(x):
        x = list(x)
        return max(x) if len(x) > 0 else -1  # -1 is default

    n_qubits = 1 + max_or_default(
        max_or_default(term[0] for term in op) for op in of_qubit_operator.terms.keys()
    )
    return n_qubits


@jit(nopython=True)
def _num_to_pauli(k):
    return ("I", "X", "Y", "Z")[k]


@jit(nopython=True)
def _pauli_to_num(p):
    if p == "X":
        return 1
    elif p == "Y":
        return 2
    elif p == "Z":
        return 3
    elif p == "I":
        return 0
    else:
        raise ValueError("p should be in 'XYZ'")


@jit(nopython=True)
def _levi_term(i, j):
    k = int(6 - i - j)  # i, j, k are permutations of (1,2,3), ijk=0 is already handled
    term = (i - j) * (j - k) * (k - i) / 2
    return _num_to_pauli(k), 1j * term


@jit(nopython=True)
def _apply_pauli_op_reduction(op1, op2):
    if op1 == op2:
        return "I", 1
    elif op1 == "I":
        return op2, 1
    elif op2 == "I":
        return op1, 1
    else:
        n1 = _pauli_to_num(op1)
        n2 = _pauli_to_num(op2)
        pauli, levi_factor = _levi_term(n1, n2)
        return pauli, levi_factor


@jit(nopython=True)
def _split_string(s):
    return [x for x in str(s)]


@jit(nopython=True)
def _make_new_pauli_string(op1, w1, op2, w2):
    """Compute the (symbolic) tensor product of two pauli strings with weights
    Args:
        op1, op2 (str): Pauli strings (e.g. IIXIIXZ).
        w1, w2 (complex): The corresponding weights

    Returns:
        new_op (str): the new pauli string (result of the tensor product)
        new_weight (complex): the weight of the pauli string

    """
    assert len(op1) == len(op2)
    op1 = _split_string(op1)
    op2 = _split_string(op2)
    o_w = [_apply_pauli_op_reduction(a, b) for a, b in zip(op1, op2)]
    new_op = [o[0] for o in o_w]
    new_weights = np.array([o[1] for o in o_w])
    new_op = "".join(new_op)
    new_weight = w1 * w2 * np.prod(new_weights)
    return new_op, new_weight


def _remove_zero_weights(op_arr, w_arr):
    if len(op_arr) == 0:
        return op_arr, w_arr
    idx_nz = ~np.isclose(w_arr, 0)
    if np.any(idx_nz):
        operators = op_arr[idx_nz]
        weights = w_arr[idx_nz]
    else:
        # convention
        operators = np.array(["I" * len(op_arr[0])])
        weights = np.array([0], dtype=w_arr.dtype)
    return operators, weights


def _reduce_pauli_string(op_arr, w_arr):
    """From a list of pauli strings, sum the weights of duplicate strings.
    Args:
        op1, op2 (str): Pauli strings (e.g. IIXIIXZ).
        w1, w2 (complex): The corresponding weights

    Returns:
        new_op (str): the new pauli string (result of the tensor product)
        new_weight (complex): the weight of the pauli string

    """
    operators_unique, idx = np.unique(op_arr, return_inverse=True)
    if len(operators_unique) == len(op_arr):
        # still remove zeros
        return _remove_zero_weights(op_arr, w_arr)
    summed_weights = np.array(
        [np.sum(w_arr[idx == i]) for i in range(len(operators_unique))]
    )
    operators, weights = _remove_zero_weights(operators_unique, summed_weights)
    return operators, weights


def _matmul(op_arr1, w_arr1, op_arr2, w_arr2):
    """(Symbolic) Tensor product of two PauliStrings
    Args:
        op_arr1, op_arr2 (np.array): Arrays operators (strings) in a PauliStrings sum
        w_arr1, w_arr2 (np.array): The corresponding weights of the operators in the sums

    Returns:
        operators (np.array): Array of the resulting operator strings
        new_weight (np.array): Array of the corresponding weights
    """

    operators = []
    weights = []
    for (op1, w1), (op2, w2) in product(zip(op_arr1, w_arr1), zip(op_arr2, w_arr2)):
        op, w = _make_new_pauli_string(op1, w1, op2, w2)
        operators.append(op)
        weights.append(w)
    operators, weights = np.array(operators), np.array(weights)
    operators, weights = _reduce_pauli_string(operators, weights)
    return operators, weights
