import numpy as np
import numba
from numba.experimental import jitclass


class LocalOperatorImpl:
    def __init__(
        self,
        acting_on,
        acting_size,
        diag_mels,
        mels,
        x_prime,
        n_conns,
        local_states,
        basis,
        constant,
        nonzero_diagonal,
    ):
        self._acting_on = acting_on
        self._acting_size = acting_size
        self._diag_mels = diag_mels
        self._mels = mels
        self._x_prime = x_prime
        self._n_conns = n_conns
        self._local_states = local_states
        self._basis = basis
        self._constant = constant
        self._nonzero_diagonal = nonzero_diagonal

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
            pad (bool): Whether to use zero-valued matrix elements in order to return all equal sections.

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        local_states = self._local_states
        basis = self._basis
        constant = self._constant
        diag_mels = self._diag_mels
        n_conns = self._n_conns
        all_mels = self._mels
        all_x_prime = self._x_prime
        acting_on = self._acting_on
        acting_size = self._acting_size
        nonzero_diagonal = self._nonzero_diagonal

        batch_size = x.shape[0]
        n_sites = x.shape[1]
        dtype = all_mels.dtype

        assert sections.shape[0] == batch_size

        n_operators = n_conns.shape[0]
        xs_n = np.empty((batch_size, n_operators), dtype=np.intp)

        tot_conn = 0
        max_conn = 0

        for b in range(batch_size):
            # diagonal element
            conn_b = 1 if nonzero_diagonal else 0

            # counting the off-diagonal elements
            for i in range(n_operators):
                acting_size_i = acting_size[i]

                xs_n[b, i] = 0
                x_b = x[b]
                x_i = x_b[acting_on[i, :acting_size_i]]
                for k in range(acting_size_i):
                    xs_n[b, i] += (
                        np.searchsorted(
                            local_states[i, acting_size_i - k - 1],
                            x_i[acting_size_i - k - 1],
                        )
                        * basis[i, k]
                    )

                conn_b += n_conns[i, xs_n[b, i]]

            tot_conn += conn_b
            sections[b] = tot_conn

            if pad:
                max_conn = max(conn_b, max_conn)

        if pad:
            tot_conn = batch_size * max_conn

        x_prime = np.empty((tot_conn, n_sites), dtype=x.dtype)
        mels = np.empty(tot_conn, dtype=dtype)

        c = 0
        for b in range(batch_size):
            c_diag = c
            x_batch = x[b]

            if nonzero_diagonal:
                mels[c_diag] = constant
                x_prime[c_diag] = np.copy(x_batch)
                c += 1

            for i in range(n_operators):

                # Diagonal part
                #  If nonzero_diagonal, this goes to c_diag = 0 ....
                # if zero_diagonal this just sets the last element to 0
                # so it's not worth it skipping it
                mels[c_diag] += diag_mels[i, xs_n[b, i]]
                n_conn_i = n_conns[i, xs_n[b, i]]

                if n_conn_i > 0:
                    sites = acting_on[i]
                    acting_size_i = acting_size[i]

                    for cc in range(n_conn_i):
                        mels[c + cc] = all_mels[i, xs_n[b, i], cc]
                        x_prime[c + cc] = np.copy(x_batch)

                        for k in range(acting_size_i):
                            x_prime[c + cc, sites[k]] = all_x_prime[
                                i, xs_n[b, i], cc, k
                            ]
                    c += n_conn_i

            if pad:
                delta_conn = max_conn - (c - c_diag)
                mels[c : c + delta_conn].fill(0)
                x_prime[c : c + delta_conn, :] = np.copy(x_batch)
                c += delta_conn
                sections[b] = c

        return x_prime, mels

    def get_conn_filtered(self, x, sections, filters):
        r"""Finds the connected elements of the Operator using only a subset of operators. Starting
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
            filters (array): Only operators op(filters[i]) are used to find the connected elements of
                        x[i].

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        local_states = self._local_states
        basis = self._basis
        constant = self._constant
        diag_mels = self._diag_mels
        n_conns = self._n_conns
        all_mels = self._mels
        all_x_prime = self._x_prime
        acting_on = self._acting_on
        acting_size = self._acting_size

        batch_size = x.shape[0]
        n_sites = x.shape[1]
        dtype = all_mels.dtype

        assert filters.shape[0] == batch_size and sections.shape[0] == batch_size

        n_operators = n_conns.shape[0]
        xs_n = np.empty((batch_size, n_operators), dtype=np.intp)

        tot_conn = 0

        for b in range(batch_size):
            # diagonal element
            tot_conn += 1

            # counting the off-diagonal elements
            i = filters[b]

            assert i < n_operators and i >= 0
            acting_size_i = acting_size[i]

            xs_n[b, i] = 0
            x_b = x[b]
            x_i = x_b[acting_on[i, :acting_size_i]]
            for k in range(acting_size_i):
                xs_n[b, i] += (
                    np.searchsorted(
                        local_states[i, acting_size_i - k - 1],
                        x_i[acting_size_i - k - 1],
                    )
                    * basis[i, k]
                )

            tot_conn += n_conns[i, xs_n[b, i]]
            sections[b] = tot_conn

        x_prime = np.empty((tot_conn, n_sites))
        mels = np.empty(tot_conn, dtype=dtype)

        c = 0
        for b in range(batch_size):
            c_diag = c
            mels[c_diag] = constant
            x_batch = x[b]
            x_prime[c_diag] = np.copy(x_batch)
            c += 1

            i = filters[b]
            # Diagonal part
            mels[c_diag] += diag_mels[i, xs_n[b, i]]
            n_conn_i = n_conns[i, xs_n[b, i]]

            if n_conn_i > 0:
                sites = acting_on[i]
                acting_size_i = acting_size[i]

                for cc in range(n_conn_i):
                    mels[c + cc] = all_mels[i, xs_n[b, i], cc]
                    x_prime[c + cc] = np.copy(x_batch)

                    for k in range(acting_size_i):
                        x_prime[c + cc, sites[k]] = all_x_prime[i, xs_n[b, i], cc, k]
                c += n_conn_i

        return x_prime, mels


_implementations = {}


def jit_implementation(dtype):
    dtype = np.dtype(dtype)
    numba_dtype = numba.typeof(dtype)[:].dtype

    spec = [
        ("_acting_on", numba.intp[:, :]),
        ("_acting_size", numba.intp[:]),
        ("_diag_mels", numba_dtype[:, :]),
        ("_mels", numba_dtype[:, :, :]),
        ("_x_prime", numba.float64[:, :, :, :]),
        ("_n_conns", numba.intp[:, :]),
        ("_local_states", numba.float64[:, :, :]),
        ("_basis", numba.int64[:, :]),
        ("_constant", numba_dtype),
        ("_nonzero_diagonal", numba.boolean),
    ]

    return jitclass(LocalOperatorImpl, spec=spec)


def get_jitted_implementation(dtype):
    dtype = np.dtype(dtype)
    if dtype not in _implementations:
        _implementations[dtype] = jit_implementation(dtype)

    return _implementations[dtype]
