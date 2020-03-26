from .abstract_operator import AbstractOperator

import numpy as _np
from numba import jit


class PyLocalOperator(AbstractOperator):
    """A custom local operator. This is a sum of an arbitrary number of operators
       acting locally on a limited set of k quantum numbers (i.e. k-local,
       in the quantum information sense).
    """

    def __init__(self, hilbert, operators, acting_on, constant=0):
        r"""
        Constructs a new ``LocalOperator`` given a hilbert space and (if
        specified) a constant level shift.

        Args:
           hilbert (netket.AbstractHilbert): Hilbert space the operator acts on.
           operators (list(numpy.array)): A list of operators, in matrix form.
           acting_on (list(numpy.array)): A list of sites, which the corresponding operators act on.
           constant (float): Level shift for operator. Default is 0.0.

        Examples:
           Constructs a ``LocalOperator`` without any operators.

           >>> from netket.graph import CustomGraph
           >>> from netket.hilbert import CustomHilbert
           >>> from netket.operator import LocalOperator
           >>> g = CustomGraph(edges=[[i, i + 1] for i in range(20)])
           >>> hi = CustomHilbert(local_states=[1, -1], graph=g)
           >>> empty_hat = LocalOperator(hi)
           >>> print(len(empty_hat.acting_on))
           0
        """
        self._constant = 0

        n_operators = len(operators)
        assert(len(acting_on) == n_operators)
        self._n_operators = n_operators

        self._acting_size = _np.zeros(n_operators, dtype=_np.intp)
        n_local_states = hilbert.local_size

        self._operators = []
        max_op_size = 0

        for i, op in enumerate(operators):
            self._operators.append(_np.asarray(
                op, dtype=_np.complex128))
            self._acting_size[i] = len(acting_on[i])
            max_op_size = max((op.shape[0], max_op_size))
            assert(op.shape[0] == n_local_states**len(acting_on[i]))

        self._max_op_size = max_op_size

        max_acting_size = self._acting_size.max()
        self._max_acting_size = max_acting_size

        self._acting_on = _np.zeros(
            (n_operators, max_acting_size), dtype=_np.intp)
        self._acting_on.fill(_np.nan)
        for i, site in enumerate(acting_on):
            acting_size = self._acting_size[i]
            self._acting_on[i, :acting_size] = _np.asarray(
                site, dtype=_np.intp)
            if(self._acting_on[i, :acting_size].max() > hilbert.size or self._acting_on[i, :acting_size].min() < 0):
                raise InvalidInputError(
                    "Operator acts on an invalid set of sites")

        self._hilbert = hilbert

        self._local_states = _np.sort(hilbert.local_states)

        self._basis = _np.zeros(
            max_acting_size, dtype=_np.int64)

        ba = 1
        for s in range(self._max_acting_size):
            self._basis[s] = ba
            ba *= self._local_states.size

        self._size = 0
        for m in self._acting_on:
            self._size = max(self._size, _np.max(m))
        self._size += 1

        self.mel_cutoff = 1.0e-6

        self._diag_mels, self._mels, self._x_prime, self._n_conns = self._init_conns()

        super().__init__()

    @property
    def hilbert(self):
        r"""AbstractHilbert: The hilbert space associated to this operator."""
        return self._hilbert

    @property
    def size(self):
        return self._size

    @property
    def mel_cutoff(self):
        r"""float: The cutoff for matrix elements.
                   Only matrix elements such that abs(O(i,i))>mel_cutoff
                   are considered """
        return self._mel_cutoff

    @mel_cutoff.setter
    def mel_cutoff(self, mel_cutoff):
        self._mel_cutoff = mel_cutoff
        assert(self.mel_cutoff >= 0)

    @staticmethod
    @jit(nopython=True)
    def _number_to_state(number, local_states, out):

        out.fill(local_states[0])
        size = out.shape[0]
        local_size = local_states.shape[0]

        ip = number
        k = size - 1
        while(ip > 0):
            out[k] = local_states[ip % local_size]
            ip = ip // local_size
            k -= 1

        return out

    def _init_conns(self):
        n_operators = self._n_operators
        max_op_size = self._max_op_size
        max_acting_size = self._max_acting_size
        epsilon = self.mel_cutoff

        diag_mels = _np.empty((n_operators, max_op_size), dtype=_np.complex128)
        mels = _np.empty(
            (n_operators, max_op_size, max_op_size), dtype=_np.complex128)
        xs_prime = _np.empty((n_operators, max_op_size,
                              max_op_size, max_acting_size))
        n_conns = _np.empty((n_operators, max_op_size), dtype=_np.intp)

        for op in range(n_operators):
            mat = self._operators[op]
            op_size = mat.shape[0]
            assert(op_size == mat.shape[1])

            diag_mel = diag_mels[op]
            mel = mels[op]
            x_prime = xs_prime[op]
            acting_size = self._acting_size[op]

            for i in range(op_size):
                diag_mel[i] = mat[i, i]
                n_conns[op, i] = 0
                for j in range(op_size):
                    if(i != j and _np.abs(mat[i, j]) > epsilon):
                        k_conn = n_conns[op, i]
                        mel[i, k_conn] = mat[i, j]
                        self._number_to_state(j, self._local_states, x_prime[i, k_conn,
                                                                             :acting_size])
                        n_conns[op, i] += 1

        return diag_mels, mels, xs_prime, n_conns

    def n_conn(self, x, out):
        r"""Return the number of states connected to x.

            Args:
                x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                            the batch of quantum numbers x.
                out (array): If None an output array is allocated.

            Returns:
                array: The number of connected states x' for each x[i].

        """
        if(out is None):
            out = _np.empty(x.shape[0], dtype=_np.int32)

        out.fill(hilbert.size + 1)

        return out

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
        return self._get_conn_kernel(x, self._local_states, self._basis,
                                     self._constant, self._diag_mels, self._n_conns,
                                     self._mels, self._x_prime, self._acting_on,
                                     self._acting_size)

    @staticmethod
    @jit(nopython=True)
    def _get_conn_kernel(x, local_states, basis, constant,
                         diag_mels, n_conns, all_mels,
                         all_x_prime, acting_on, acting_size):

        n_operators = n_conns.shape[0]
        xs_n = _np.empty(n_operators, dtype=_np.intp)
        tot_conn = 1

        for i in range(n_operators):
            acting_size_i = acting_size[i]

            xs_n[i] = 0

            x_i = x[acting_on[i, :acting_size_i]]
            for k in range(acting_size_i):
                xs_n[i] += _np.searchsorted(local_states,
                                            x_i[acting_size_i - k - 1]) * basis[k]

            tot_conn += n_conns[i, xs_n[i]]

        mels = _np.empty(tot_conn, dtype=_np.complex128)
        x_prime = _np.empty((tot_conn, x.shape[0]))

        mels[0] = constant
        x_prime[0] = _np.copy(x)
        c = 1

        for i in range(n_operators):

            # Diagonal part
            mels[0] += diag_mels[i, xs_n[i]]
            n_conn_i = n_conns[i, xs_n[i]]

            if(n_conn_i > 0):
                sites = acting_on[i]
                acting_size_i = acting_size[i]

                for cc in range(n_conn_i):
                    mels[c + cc] = all_mels[i, xs_n[i], cc]
                    x_prime[c + cc] = _np.copy(x)

                    for k in range(acting_size_i):
                        x_prime[c + cc, sites[k]
                                ] = all_x_prime[i, xs_n[i], cc, k]
                c += n_conn_i

        return x_prime, mels

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

        return NotImplementedError
