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

from netket.hilbert import Qubit

from ._abstract_operator import AbstractOperator


class PauliStrings(AbstractOperator):
    """A Hamiltonian consisiting of the sum of products of Pauli operators."""

    def __init__(
        self,
        operators: List[str],
        weights: List[Union[float, complex]],
        cutoff: float = 1.0e-10,
        dtype: DType = complex,
    ):
        """
        Constructs a new ``PauliStrings`` operator given a set of Pauli operators.

        Args:
           operators (list(string)): A list of Pauli operators in string format, e.g. ['IXX', 'XZI'].
           weights: A list of amplitudes of the corresponding Pauli operator.
           cutoff (float): a cutoff to remove small matrix elements

        Examples:
           Constructs a new ``PauliStrings`` operator X_0*X_1 + 3.*Z_0*Z_1.

           >>> import netket as nk
           >>> op = nk.operator.PauliStrings(operators=['XX','ZZ'], weights=[1,3])
           >>> op.hilbert.size
           2
        """
        if len(operators) == 0:
            raise ValueError("No Pauli operators passed.")

        if len(weights) != len(operators):
            raise ValueError("weights should have the same length as operators.")

        if not np.isscalar(cutoff) or cutoff < 0:
            raise ValueError("invalid cutoff in PauliStrings.")

        _n_qubits = len(operators[0])
        consistent = all(len(op) == _n_qubits for op in operators)
        if not consistent:
            raise ValueError("Pauli strings have inhomogeneous lengths.")

        def valid_match(strg, search=re.compile(r"^[XYZI]+$").search):
            return bool(search(strg))

        consistent = all(valid_match(op) for op in operators)
        if not consistent:
            raise ValueError(
                """Operators in string must be one of
                the Pauli operators X,Y,Z, or the identity I"""
            )

        self._n_qubits = _n_qubits
        super().__init__(Qubit(_n_qubits))

        n_operators = len(operators)

        self._cutoff = cutoff
        b_weights = np.asarray(weights, dtype=dtype)
        self._is_hermitian = np.allclose(b_weights.imag, 0.0)

        b_to_change = [] * n_operators
        b_z_check = [] * n_operators

        acting = {}

        def find_char(s, ch):
            return [i for i, ltr in enumerate(s) if ltr == ch]

        def append(key, k):
            # convert list to tuple
            key = tuple(key)
            if key in acting:
                acting[key].append(k)
            else:
                acting[key] = [k]

        _n_z_check_max = 0

        for i, op in enumerate(operators):
            b_to_change = []
            b_z_check = []
            b_weights = weights[i]

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
        _sites = np.empty((n_operators, _n_qubits), dtype=np.intp)
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

        self._x_prime_max = np.empty((n_operators, _n_qubits))
        self._mels_max = np.empty((n_operators), dtype=dtype)
        self._n_operators = n_operators
        self._dtype = dtype

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
        return self._n_operators

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
        pad=False,
    ):

        x_prime = np.zeros((x.shape[0] * max_conn, x_prime.shape[1]))
        mels = np.zeros((x.shape[0] * max_conn), dtype=mels.dtype)

        n_c = 0
        for b in range(x.shape[0]):
            xb = x[b]
            for i in range(sites.shape[0]):
                mel = 0.0
                for j in range(n_op[i]):
                    if nz_check[i, j] > 0:
                        to_check = z_check[i, j, : nz_check[i, j]]
                        n_z = np.count_nonzero(xb[to_check] == 1)
                    else:
                        n_z = 0

                    mel += weights[i, j] * (-1.0) ** n_z

                if abs(mel) > cutoff:
                    x_prime[n_c] = np.copy(xb)
                    for site in sites[i, : ns[i]]:
                        x_prime[n_c, site] = 1 - x_prime[n_c, site]
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
            pad,
        )

    def _get_conn_flattened_closure(self):
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
            )

        return jit(nopython=True)(gccf_fun)
