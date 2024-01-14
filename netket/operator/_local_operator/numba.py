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

from typing import TYPE_CHECKING

import numpy as np
import numba

from netket.errors import concrete_or_error, NumbaOperatorGetConnDuringTracingError


from .compile_helpers import pack_internals
from .base import LocalOperatorBase

if TYPE_CHECKING:
    from .jax import LocalOperatorJax


class LocalOperator(LocalOperatorBase):
    """A custom local operator. This is a sum of an arbitrary number of operators
    acting locally on a limited set of k quantum numbers (i.e. k-local,
    in the quantum information sense).
    """

    __module__ = "netket.operator"

    def _setup(self, force: bool = False):
        """Analyze the operator strings and precompute arrays for get_conn inference"""
        if force or not self._initialized:
            data = pack_internals(
                self.hilbert,
                self._operators_dict,
                self.constant,
                self.dtype,
                self.mel_cutoff,
            )

            self._acting_on = data["acting_on"]
            self._acting_size = data["acting_size"]
            self._diag_mels = data["diag_mels"]
            self._mels = data["mels"]
            self._x_prime = data["x_prime"]
            self._n_conns = data["n_conns"]
            self._local_states = data["local_states"]
            self._basis = data["basis"]
            self._nonzero_diagonal = data["nonzero_diagonal"]
            self._max_conn_size = data["max_conn_size"]
            self._initialized = True

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
        self._setup()

        x = concrete_or_error(
            np.asarray,
            x,
            NumbaOperatorGetConnDuringTracingError,
            self,
        )

        return self._get_conn_flattened_kernel(
            x,
            sections,
            self._local_states,
            self._basis,
            self._constant,
            self._diag_mels,
            self._n_conns,
            self._mels,
            self._x_prime,
            self._acting_on,
            self._acting_size,
            self._nonzero_diagonal,
            pad,
        )

    def _get_conn_flattened_closure(self):
        self._setup()
        _local_states = self._local_states
        _basis = self._basis
        _constant = self._constant
        _diag_mels = self._diag_mels
        _n_conns = self._n_conns
        _mels = self._mels
        _x_prime = self._x_prime
        _acting_on = self._acting_on
        _acting_size = self._acting_size
        # workaround my painfully discovered Numba#6979 (cannot use numpy bools in closures)
        _nonzero_diagonal = bool(self._nonzero_diagonal)

        fun = self._get_conn_flattened_kernel

        def gccf_fun(x, sections):
            return fun(
                x,
                sections,
                _local_states,
                _basis,
                _constant,
                _diag_mels,
                _n_conns,
                _mels,
                _x_prime,
                _acting_on,
                _acting_size,
                _nonzero_diagonal,
            )

        return numba.jit(nopython=True)(gccf_fun)

    @staticmethod
    @numba.jit(nopython=True)
    def _get_conn_flattened_kernel(
        x,
        sections,
        local_states,
        basis,
        constant,
        diag_mels,
        n_conns,
        all_mels,
        all_x_prime,
        acting_on,
        acting_size,
        nonzero_diagonal,
        pad=False,
    ):
        batch_size = x.shape[0]
        n_sites = x.shape[1]
        dtype = all_mels.dtype

        # TODO remove this line when numba 0.53 is dropped 0.54 is minimum version
        # workaround a bug in Numba arising when NUMBA_BOUNDSCHECK=1
        constant = constant.item()

        assert sections.shape[0] == batch_size

        n_operators = n_conns.shape[0]
        # array to store the row index
        xs_n = np.empty((batch_size, n_operators), dtype=np.intp)

        tot_conn = 0
        max_conn = 0

        for b in range(batch_size):
            # diagonal element
            conn_b = 1 if nonzero_diagonal else 0

            # counting the off-diagonal elements
            for i in range(n_operators):
                acting_size_i = acting_size[i]
                # compute the number (row index) from the local states
                # (here we do the inverse of _number_to_state from
                # compile_helpers.py, so this is essentially _state_to_number)
                xs_n[b, i] = 0
                x_b = x[b]
                x_i = x_b[acting_on[i, :acting_size_i]]
                # iterate over sites the current operator is acting on
                for k in range(acting_size_i):
                    # compute
                    xs_n[b, i] += (
                        np.searchsorted(
                            local_states[i, acting_size_i - k - 1],
                            x_i[acting_size_i - k - 1],
                        )
                        * basis[i, k]
                    )

                # sum the number of off-diagonal connected elements
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
                if nonzero_diagonal:
                    mels[c_diag] += diag_mels[i, xs_n[b, i]]

                # get the number of connected elements for the current operator
                # at the rows index corresponding to the state of x at the sites
                # the operator is acting on
                n_conn_i = n_conns[i, xs_n[b, i]]

                if n_conn_i > 0:
                    sites = acting_on[i]
                    acting_size_i = acting_size[i]

                    for cc in range(n_conn_i):  # iterate over compressed nonzero cols
                        # get the nonzero mels of the current row
                        mels[c + cc] = all_mels[i, xs_n[b, i], cc]
                        x_prime[c + cc] = np.copy(x_batch)
                        # set the changed local states of the sites the operator
                        # is acting on
                        # it is stored in all_x_prime, where we select the row
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
        self._setup()

        x = concrete_or_error(
            np.asarray,
            x,
            NumbaOperatorGetConnDuringTracingError,
            self,
        )

        return self._get_conn_filtered_kernel(
            x,
            sections,
            self._local_states,
            self._basis,
            self._constant,
            self._diag_mels,
            self._n_conns,
            self._mels,
            self._x_prime,
            self._acting_on,
            self._acting_size,
            filters,
        )

    @staticmethod
    @numba.jit(nopython=True)
    def _get_conn_filtered_kernel(
        x,
        sections,
        local_states,
        basis,
        constant,
        diag_mels,
        n_conns,
        all_mels,
        all_x_prime,
        acting_on,
        acting_size,
        filters,
    ):
        batch_size = x.shape[0]
        n_sites = x.shape[1]
        dtype = all_mels.dtype

        assert filters.shape[0] == batch_size and sections.shape[0] == batch_size

        # TODO remove this line when numba 0.53 is dropped 0.54 is minimum version
        # workaround a bug in Numba arising when NUMBA_BOUNDSCHECK=1
        constant = constant.item()

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

    def to_jax_operator(self) -> "LocalOperatorJax":  # noqa: F821
        """
        Returns the jax-compatible version of this operator, which is an
        instance of :class:`netket.operator.LocalOperatorJax`.
        """
        from .jax import LocalOperatorJax

        return LocalOperatorJax(
            self.hilbert,
            self.operators,
            self.acting_on,
            self.constant,
            dtype=self.dtype,
        )
