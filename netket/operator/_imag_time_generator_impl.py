import numbers
from typing import Union, Tuple, List, Optional
from numpy.typing import DTypeLike, ArrayLike

import numpy as np
import numba
from numba import jit
from numba.experimental import jitclass


class ImagTimeGeneratorImpl:
    def __init__(self, hamiltonian, max_conn_size, xprime_f, mels_f):
        self._hamiltonian = hamiltonian
        self._xprime_f = xprime_f
        self._mels_f = mels_f
        self._max_conn_size = max_conn_size

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
        batch_size = x.shape[0]
        N = x.shape[1] // 2

        hamiltonian = self._hamiltonian

        if self._xprime_f.shape[0] < self._max_conn_size * batch_size:
            # refcheck=False because otherwise this errors when testing
            self._xprime_f = np.empty(
                (self._max_conn_size * batch_size, self._xprime_f.shape[1]),
                dtype=self._xprime_f.dtype,
            )
            self._mels_f = np.empty(
                self._max_conn_size * batch_size, dtype=self._mels_f.dtype
            )

        xs = self._xprime_f
        mels = self._mels_f

        # Separate row and column inputs
        xr, xc = x[:, 0:N], x[:, N : 2 * N]

        sections_r = np.empty(batch_size, dtype=np.int64)
        sections_c = np.empty(batch_size, dtype=np.int64)

        xr_prime, mels_r = hamiltonian.get_conn_flattened(xr, sections_r)
        xc_prime, mels_c = hamiltonian.get_conn_flattened(xc, sections_c)

        if pad:
            # if else to accomodate for batches of 1 element, because
            # sections don't start from 0-index...
            # TODO: if we ever switch to 0-indexing, remove this.
            if batch_size > 1:
                max_conns_r = np.max(np.diff(sections_r))
                max_conns_c = np.max(np.diff(sections_c))
            else:
                max_conns_r = sections_r[0]
                max_conns_c = sections_c[0]

        if pad:
            pad = max_conns_r + max_conns_c
        else:
            pad = 0

        #

        xs[:, :] = 0
        mels[:] = 0

        sec = 0
        off = 0

        n_hr_i = 0
        n_hc_i = 0

        for i in range(batch_size):
            off_i = off
            n_hr_f = sections_r[i]
            n_hr = n_hr_f - n_hr_i
            xs[off : off + n_hr, 0:N] = xr_prime[n_hr_i:n_hr_f, :]
            xs[off : off + n_hr, N : 2 * N] = xc[i, :]
            mels[off : off + n_hr] = -mels_r[n_hr_i:n_hr_f]
            off += n_hr
            n_hr_i = n_hr_f

            n_hc_f = sections_c[i]
            n_hc = n_hc_f - n_hc_i
            xs[off : off + n_hc, N : 2 * N] = xc_prime[n_hc_i:n_hc_f, :]
            xs[off : off + n_hc, 0:N] = xr[i, :]
            mels[off : off + n_hc] = -mels_c[n_hc_i:n_hc_f]
            off += n_hc
            n_hc_i = n_hc_f

            if pad != 0:
                n_entries = off - off_i
                mels[off : off + n_entries] = 0
                off = (i + 1) * pad

            sections[i] = off

        return np.copy(xs[0:off, :]), np.copy(mels[0:off])


_implementations = {}


def jit_implementation(signature):
    T, dtype = signature
    dtype = np.dtype(dtype)
    numba_dtype = numba.typeof(dtype)[:].dtype

    spec = [
        ("_max_conn_size", numba.intp),
        ("_hamiltonian", T),
        ("_xprime_f", numba.float64[:, :]),
        ("_mels_f", numba_dtype[:]),
    ]

    return jitclass(ImagTimeGeneratorImpl, spec=spec)


def get_jitted_implementation(hamiltonian, dtype):
    T_ham = numba.typeof(hamiltonian)
    dtype = np.dtype(dtype)

    sig = (T_ham, dtype)
    if sig not in _implementations:
        _implementations[sig] = jit_implementation(sig)

    return _implementations[sig]
