from ._abstract_operator import AbstractOperator
from ..hilbert import Qubit
from ..graph import Edgeless

import numpy as _np
from numba import jit
import re


class PauliStrings(AbstractOperator):
    """A Hamiltonian consisiting of the sum of products of Pauli operators."""

    def __init__(self, operators, weights, cutoff=1.0e-10):
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

        if not _np.isscalar(cutoff) or cutoff < 0:
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

        graph = Edgeless(_n_qubits)
        self._n_qubits = _n_qubits
        self._hilbert = Qubit(graph)

        n_operators = len(operators)

        self._cutoff = cutoff
        b_weights = _np.asarray(weights, dtype=_np.complex128)

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
                b_weights *= (1.0j) ** (len(y_ops))
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
        _sites = _np.empty((n_operators, _n_qubits), dtype=_np.intp)
        _ns = _np.empty((n_operators), dtype=_np.intp)
        _n_op = _np.empty(n_operators, dtype=_np.intp)
        _weights = _np.empty((n_operators, _n_op_max), dtype=_np.complex128)
        _nz_check = _np.empty((n_operators, _n_op_max), dtype=_np.intp)
        _z_check = _np.empty((n_operators, _n_op_max, _n_z_check_max), dtype=_np.intp)

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

        self._x_prime_max = _np.empty((n_operators, _n_qubits))
        self._mels_max = _np.empty((n_operators), dtype=_np.complex128)
        self._n_operators = n_operators

        super().__init__()

    @property
    def hilbert(self):
        r"""AbstractHilbert: The hilbert space associated to this operator."""
        return self._hilbert

    @property
    def size(self):
        return self._n_qubits

    def get_conn(self, x):
        r"""Finds the connected elements of the Operator. Starting
            from a given quantum number x, it finds all other quantum numbers x' such
            that the matrix element :math:`O(x,x')` is different from zero. In general there
            will be several different connected states x' satisfying this
            condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

            This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

            Args:
                x (array): An array of shape (hilbert.size) containing the quantum numbers x.

            Returns:
                matrix: The connected states x' of shape (N_connected,hilbert.size)
                array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """
        return self._flattened_kernel(
            _np.atleast_2d(x),
            self._x_prime_max,
            _np.ones(1),
            self._mels_max,
            self._sites,
            self._ns,
            self._n_op,
            self._weights,
            self._nz_check,
            self._z_check,
            self._cutoff,
            self._n_operators,
        )

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(
        x,
        x_prime,
        sections,
        mels,
        sites,
        ns,
        n_op,
        weights,
        nz_check,
        z_check,
        cutoff,
        max_conn,
    ):
        if x_prime.shape[0] < x.shape[0] * max_conn:
            x_prime = _np.empty((x.shape[0] * max_conn, x_prime.shape[1]))
            mels = _np.empty((x.shape[0] * max_conn), dtype=_np.complex128)

        n_c = 0
        for b in range(x.shape[0]):
            xb = x[b]
            for i in range(sites.shape[0]):
                mel = 0.0
                for j in range(n_op[i]):
                    if nz_check[i, j] > 0:
                        to_check = z_check[i, j, : nz_check[i, j]]
                        n_z = _np.count_nonzero(xb[to_check] == 1)
                    else:
                        n_z = 0

                    mel += weights[i, j] * (-1.0) ** n_z

                if abs(mel) > cutoff:
                    x_prime[n_c] = _np.copy(xb)
                    for site in sites[i, : ns[i]]:
                        x_prime[n_c, site] = 1 - x_prime[n_c, site]
                    mels[n_c] = mel
                    n_c += 1
            sections[b] = n_c
        return _np.copy(x_prime), _np.copy(mels)

    def get_conn_flattened(self, x, sections):
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
            self._x_prime_max,
            sections,
            self._mels_max,
            self._sites,
            self._ns,
            self._n_op,
            self._weights,
            self._nz_check,
            self._z_check,
            self._cutoff,
            self._n_operators,
        )
