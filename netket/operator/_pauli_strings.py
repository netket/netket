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
from typing import List, Union
from netket.utils.types import DType

import numpy as np
from numba import jit
from itertools import product

from netket.hilbert import Qubit, AbstractHilbert

from ._discrete_operator import DiscreteOperator

valid_pauli_regex = re.compile(r"^[XYZI]+$")


class PauliStrings(DiscreteOperator):
    """A Hamiltonian consisiting of the sum of products of Pauli operators."""

    def __init__(
        self,
        hilbert: AbstractHilbert,
        operators: List[str] = None,
        weights: List[Union[float, complex]] = None,
        *,
        cutoff: float = 1.0e-10,
        dtype: DType = complex,
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

        if not isinstance(hilbert, AbstractHilbert):
            # if first argument is not Hilbert, then shift all arguments by one
            hilbert, operators, weights = None, hilbert, operators

        if operators is None:
            raise ValueError(
                "None valued operators passed. (Might arised when passing None valued hilbert explicitly)"
            )

        if len(operators) == 0:
            raise ValueError("No Pauli operators passed.")

        if weights is None:
            # default weight is 1
            weights = [True for i in operators]

        if len(weights) != len(operators):
            raise ValueError("weights should have the same length as operators.")

        if not np.isscalar(cutoff) or cutoff < 0:
            raise ValueError("invalid cutoff in PauliStrings.")

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

        super().__init__(hilbert)
        if self.hilbert.local_size != 2:
            raise ValueError(
                "PauliStrings only work for local hilbert size 2 where PauliMatrices are defined"
            )

        self._cutoff = cutoff
        b_weights = np.asarray(weights, dtype=dtype)
        self._is_hermitian = np.allclose(b_weights.imag, 0.0)

        self._orig_operators = np.array(operators, dtype=str)
        self._orig_weights = np.array(weights, dtype=dtype)
        self._dtype = dtype

        self._initialized = False

    @staticmethod
    def identity(hilbert: AbstractHilbert, **kwargs):
        operators = ("I" * hilbert.size,)
        weights = (1.0,)
        return PauliStrings(hilbert, operators, weights, **kwargs)

    def _setup(self, force=False):
        """Analyze the operator strings and precompute arrays for get_conn inference"""
        if force or not self._initialized:

            dtype = self._dtype
            n_operators = len(self._orig_operators)
            hilb_size = self.hilbert.size

            b_to_change = [] * n_operators
            b_z_check = [] * n_operators

            acting = {}

            def find_char(s, ch):
                return [i for i, ltr in enumerate(s) if ltr == ch]

            def append(key, k):
                # convert list to tuple
                key = tuple(sorted(key))  # order of X and Y does not matter
                if key in acting:
                    acting[key].append(k)
                else:
                    acting[key] = [k]

            _n_z_check_max = 0

            for i, op in enumerate(self._orig_operators):
                b_to_change = []
                b_z_check = []
                b_weights = self._orig_weights[i]

                x_ops = find_char(op, "X")
                if len(x_ops):
                    b_to_change += x_ops

                y_ops = find_char(op, "Y")
                if len(y_ops):
                    b_to_change += y_ops
                    b_weights *= (-1.0j) ** (len(y_ops))
                    b_z_check += y_ops

                z_ops = find_char(op, "Z")
                if len(z_ops):
                    b_z_check += z_ops

                _n_z_check_max = max(_n_z_check_max, len(b_z_check))
                append(b_to_change, (b_weights, b_z_check))

            # now group together operators with same final state
            n_operators = len(acting)
            _n_op_max = max(
                list(map(lambda x: len(x), list(acting.values()))), default=n_operators
            )

            # unpacking the dictionary into fixed-size arrays
            _sites = np.empty((n_operators, hilb_size), dtype=np.intp)
            _ns = np.empty((n_operators), dtype=np.intp)
            _n_op = np.empty(n_operators, dtype=np.intp)
            _weights = np.empty((n_operators, _n_op_max), dtype=dtype)
            _nz_check = np.empty((n_operators, _n_op_max), dtype=np.intp)
            _z_check = np.empty((n_operators, _n_op_max, _n_z_check_max), dtype=np.intp)

            for i, act in enumerate(acting.items()):
                sites = act[0]
                nsi = len(sites)
                _sites[i, :nsi] = sites
                _ns[i] = nsi
                values = act[1]
                _n_op[i] = len(values)
                for j in range(_n_op[i]):
                    _weights[i, j] = values[j][0]
                    _nz_check[i, j] = len(values[j][1])
                    _z_check[i, j, : _nz_check[i, j]] = values[j][1]

            self._sites = _sites
            self._ns = _ns
            self._n_op = _n_op
            self._weights = _weights
            self._nz_check = _nz_check
            self._z_check = _z_check

            self._x_prime_max = np.empty((n_operators, hilb_size))
            self._mels_max = np.empty((n_operators), dtype=dtype)
            self._n_operators = n_operators

            self._local_states = np.array(self.hilbert.local_states)

            self._initialized = True

    @staticmethod
    def from_openfermion(
        hilbert: AbstractHilbert,
        of_qubit_operator: "openfermion.ops.QubitOperator" = None,  # noqa: F821
        *,
        n_qubits: int = None,
    ):
        r"""
        Converts an openfermion QubitOperator into a netket PauliStrings.
        The hilbert first argument can be dropped, see __init__ for details and default value

        Args:
            hilbert (optional): hilbert of the resulting PauliStrings object
            of_qubit_operator (required): openfermion.ops.QubitOperator object
            n_qubits (int): total number of qubits in the system, default None means inferring it from the QubitOperator. Argument is ignored when hilbert is given.

        Returns:
            A PauliStrings object.
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
            # we always start counting from 0, so we only determine the maximum location
            n_qubits = (
                max(
                    max(term[0] for term in op) for op in of_qubit_operator.terms.keys()
                )
                + 1
            )
        for operator, weight in of_qubit_operator.terms.items():  # gives dict
            s = ["I"] * n_qubits
            for loc, op in operator:
                assert (
                    loc < n_qubits
                ), "operator index {} is longer than n_qubits={}".format(loc, n_qubits)
                s[loc] = op
            operators.append("".join(s))
            weights.append(weight)

        ps_args = (operators, weights)
        if hilbert is not None:
            ps_args = (hilbert,) + ps_args
        return PauliStrings(*ps_args)

    @property
    def dtype(self) -> DType:
        """The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩."""
        return self._dtype

    @property
    def is_hermitian(self) -> bool:
        """Returns true if this operator is hermitian."""
        return self._is_hermitian

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        # 1 connection for every operator X, Y, Z...
        self._setup()
        return self._n_operators

    def __repr__(self):
        print_list = []
        for op, w in zip(self._orig_operators, self._orig_weights):
            print_list.append("    {} : {}".format(op, str(w)))
        s = "PauliStrings(hilbert={}, n_strings={}, dict(operators:weights)=\n{}\n)".format(
            self.hilbert, len(self._orig_operators), ",\n".join(print_list)
        )
        return s

    def _op__matmul__(self, other):
        if not isinstance(other, PauliStrings):
            return NotImplementedError
        if not self.hilbert == other.hilbert:
            raise ValueError(
                f"Can only multiply identical hilbert spaces (got A@B, A={self.hilbert}, B={other.hilbert})"
            )
        operators, weights = _matmul(
            self._orig_operators,
            self._orig_weights,
            other._orig_operators,
            other._orig_weights,
        )
        return PauliStrings(
            self.hilbert,
            operators,
            weights,
            cutoff=max(self._cutoff, other._cutoff),
            dtype=self.dtype,
        )

    def __rmul__(self, scalar):
        return self * scalar

    def __mul__(self, scalar):
        if not np.issubdtype(type(scalar), np.number):
            raise NotImplementedError
        weights = self._orig_weights * scalar
        return PauliStrings(
            self.hilbert,
            self._orig_operators,
            weights,
            dtype=self.dtype,
            cutoff=self._cutoff,
        )

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        if np.issubdtype(type(other), np.number):
            if other != 0.0:
                # adding a constant = adding IIII...III with weight being the constant
                return self + PauliStrings.identity(self.hilbert) * other
            return self
        if not isinstance(other, PauliStrings):
            raise NotImplementedError
        if not self.hilbert == other.hilbert:
            raise ValueError(
                f"Can only add identical hilbert spaces (got A+B, A={self.hilbert}, B={other.hilbert})"
            )
        operators = np.concatenate((self._orig_operators, other._orig_operators))
        weights = np.concatenate((self._orig_weights, other._orig_weights))
        operators, weights = _reduce_pauli_string(operators, weights)
        return PauliStrings(
            self.hilbert,
            operators,
            weights,
            dtype=self.dtype,
            cutoff=self._cutoff,
        )

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(
        x,
        sections,
        x_prime,
        mels,
        sites,
        ns,
        n_op,
        weights,
        nz_check,
        z_check,
        cutoff,
        max_conn,
        local_states,
        pad=False,
    ):
        x_prime = np.empty((x.shape[0] * max_conn, x_prime.shape[1]), dtype=x.dtype)
        mels = np.zeros((x.shape[0] * max_conn), dtype=mels.dtype)
        state_1 = local_states[-1]

        n_c = 0
        for b in range(x.shape[0]):
            xb = x[b]
            # initialize
            x_prime[b * max_conn : (b + 1) * max_conn, :] = np.copy(xb)

            for i in range(sites.shape[0]):
                mel = 0.0
                for j in range(n_op[i]):
                    if nz_check[i, j] > 0:
                        to_check = z_check[i, j, : nz_check[i, j]]
                        n_z = np.count_nonzero(xb[to_check] == state_1)
                    else:
                        n_z = 0

                    mel += weights[i, j] * (-1.0) ** n_z

                if abs(mel) > cutoff:
                    x_prime[n_c] = np.copy(xb)
                    for site in sites[i, : ns[i]]:
                        new_state_idx = int(x_prime[n_c, site] == local_states[0])
                        x_prime[n_c, site] = local_states[new_state_idx]
                    mels[n_c] = mel
                    n_c += 1

            if pad:
                n_c = (b + 1) * max_conn

            sections[b] = n_c
        return x_prime[:n_c], mels[:n_c]

    def get_conn_flattened(self, x, sections, pad=False):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            sections (array): An array of size (batch_size) useful to unflatten
                        the output of this function.
                        See numpy.split for the meaning of sections.

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """
        self._setup()
        x = np.array(x)
        assert (
            x.shape[-1] == self.hilbert.size
        ), "size of hilbert space does not match size of x"
        return self._flattened_kernel(
            x,
            sections,
            self._x_prime_max,
            self._mels_max,
            self._sites,
            self._ns,
            self._n_op,
            self._weights,
            self._nz_check,
            self._z_check,
            self._cutoff,
            self._n_operators,
            self._local_states,
            pad,
        )

    def _get_conn_flattened_closure(self):
        self._setup()
        _x_prime_max = self._x_prime_max
        _mels_max = self._mels_max
        _sites = self._sites
        _ns = self._ns
        _n_op = self._n_op
        _weights = self._weights
        _nz_check = self._nz_check
        _z_check = self._z_check
        _cutoff = self._cutoff
        _n_operators = self._n_operators
        fun = self._flattened_kernel
        _local_states = self._local_states

        def gccf_fun(x, sections):
            return fun(
                x,
                sections,
                _x_prime_max,
                _mels_max,
                _sites,
                _ns,
                _n_op,
                _weights,
                _nz_check,
                _z_check,
                _cutoff,
                _n_operators,
                _local_states,
            )

        return jit(nopython=True)(gccf_fun)


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
