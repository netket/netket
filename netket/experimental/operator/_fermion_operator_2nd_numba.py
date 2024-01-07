# Copyright 2022 The NetKet Authors - All rights reserved.
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

import numpy as np
import numba

from netket.utils.types import DType
from netket.errors import concrete_or_error, NumbaOperatorGetConnDuringTracingError

from ._fermion_operator_2nd_utils import _is_diag_term, OperatorDict
from ._fermion_operator_2nd_base import FermionOperator2ndBase


class FermionOperator2nd(FermionOperator2ndBase):
    r"""
    A fermionic operator in :math:`2^{nd}` quantization, using Numba
    for indexing.

    .. warning::

        This class is not a Pytree, so it cannot be used inside of jax-transformed
        functions like `jax.grad` or `jax.jit`.

        The standard usage is to index into the operator from outside the jax
        function transformation and pass the results to the jax-transformed functions.

        To use this operator inside of a jax function transformation, convert it to a
        jax operator (class:`netket.experimental.operator.FermionOperator2ndJax`) by using
        :meth:`netket.experimental.operator.FermionOperator2nd.to_jax_operator()`.

    When using native (experimental) sharding, or when working with GPUs,
    we reccomend using the Jax implementations of the operators for potentially
    better performance.
    """

    def _setup(self, force: bool = False):
        """Analyze the operator strings and precompute arrays for get_conn inference"""
        if force or not self._initialized:
            # following lists will be used to compute matrix elements
            # they are filled in _add_term
            out = _pack_internals(self._operators, self._dtype)
            (
                self._orb_idxs,
                self._daggers,
                self._numba_weights,
                self._diag_idxs,
                self._off_diag_idxs,
                self._term_split_idxs,
                _collected_constant,
            ) = out
            self._constant += _collected_constant

            self._max_conn_size = 0
            if not _isclose(self._constant, 0) or len(self._diag_idxs) > 0:
                self._max_conn_size += 1
            # the following could be reduced further
            self._max_conn_size += len(self._off_diag_idxs)

            self._initialized = True

    def to_jax_operator(self) -> "FermionOperator2ndJax":  # noqa: F821
        """
        Returns the jax version of this operator, which is an
        instance of :class:`netket.experimental.operator.FermionOperator2ndJax`.
        """
        from ._fermion_operator_2nd_jax import FermionOperator2ndJax

        new_op = FermionOperator2ndJax(
            self.hilbert, constant=self._constant, dtype=self.dtype
        )
        new_op._operators = self._operators.copy()
        return new_op

    def _get_conn_flattened_closure(self):
        self._setup()
        _max_conn_size = self.max_conn_size
        _orb_idxs = self._orb_idxs
        _daggers = self._daggers
        _weights = self._numba_weights
        _diag_idxs = self._diag_idxs
        _off_diag_idxs = self._off_diag_idxs
        _term_split_idxs = self._term_split_idxs

        _constant = self._constant
        fun = self._flattened_kernel

        def gccf_fun(x, sections):
            return fun(
                x,
                sections,
                _max_conn_size,
                _orb_idxs,
                _daggers,
                _weights,
                _diag_idxs,
                _off_diag_idxs,
                _term_split_idxs,
                _constant,
            )

        return numba.jit(nopython=True)(gccf_fun)

    def get_conn_flattened(self, x, sections, pad=False):
        r"""Finds the connected elements of the Operator.

        Starting from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x: A matrix of shape (batch_size,hilbert.size) containing
                the batch of quantum numbers x.
            sections: An array of size (batch_size) useful to unflatten
                the output of this function.
                See numpy.split for the meaning of sections.

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

        assert (
            x.shape[-1] == self.hilbert.size
        ), "size of hilbert space does not match size of x"
        return self._flattened_kernel(
            x,
            sections,
            self.max_conn_size,
            self._orb_idxs,
            self._daggers,
            self._numba_weights,
            self._diag_idxs,
            self._off_diag_idxs,
            self._term_split_idxs,
            self._constant,
            pad,
        )

    @staticmethod
    @numba.jit(nopython=True)
    def _flattened_kernel(  # pragma: no cover
        x,
        sections,
        max_conn,
        orb_idxs,
        daggers,
        weights,
        diag_idxs,
        off_diag_idxs,
        term_split_idxs,
        constant,
        pad=False,
    ):
        x_prime = np.empty((x.shape[0] * max_conn, x.shape[1]), dtype=x.dtype)
        mels = np.zeros((x.shape[0] * max_conn), dtype=weights.dtype)

        # do not split at the last one (gives empty array)
        term_split_idxs = term_split_idxs[:-1]
        orb_idxs_list = np.split(orb_idxs, term_split_idxs)
        daggers_list = np.split(daggers, term_split_idxs)

        has_constant = not _isclose(constant, 0.0)

        # loop over the batch dimension
        n_c = 0
        for b in range(x.shape[0]):
            xb = x[b, :]

            # we can already fill up with default values
            if pad:
                x_prime[b * max_conn : (b + 1) * max_conn, :] = np.copy(xb)

            non_zero_diag = False
            if has_constant:
                non_zero_diag = True
                x_prime[n_c, :] = np.copy(xb)
                mels[n_c] += constant

            # first do the diagonal terms, they all generate just 1 term
            for term_idx in diag_idxs:
                mel = weights[term_idx]
                xt = np.copy(xb)
                has_xp = True
                for orb_idx, dagger in zip(
                    orb_idxs_list[term_idx], daggers_list[term_idx]
                ):
                    _, mel, op_has_xp = _apply_operator(xt, orb_idx, dagger, mel)
                    if not op_has_xp:
                        has_xp = False
                        continue
                if has_xp:
                    x_prime[n_c, :] = np.copy(xb)  # should be untouched
                    mels[n_c] += mel

                non_zero_diag = non_zero_diag or has_xp

            # end of the diagonal terms
            if non_zero_diag:
                n_c += 1

            # now do the off-diagonal terms
            for term_idx in off_diag_idxs:
                mel = weights[term_idx]
                xt = np.copy(xb)
                has_xp = True
                for orb_idx, dagger in zip(
                    orb_idxs_list[term_idx], daggers_list[term_idx]
                ):
                    xt, mel, op_has_xp = _apply_operator(xt, orb_idx, dagger, mel)
                    if not op_has_xp:  # detect zeros
                        has_xp = False
                        continue
                if has_xp:
                    x_prime[n_c, :] = np.copy(xt)  # should be different
                    mels[n_c] += mel
                    n_c += 1

            # end of this sample
            if pad:
                n_c = (b + 1) * max_conn

            sections[b] = n_c

        if pad:
            return x_prime, mels
        else:
            return x_prime[:n_c], mels[:n_c]


def _pack_internals(operators: OperatorDict, dtype: DType):
    """
    Create the internal structures to compute the matrix elements
    Processes and adds a single term such that we can compute its matrix elements, in tuple format ((1,1), (2,0))
    """
    # properties of single-fermion operators, e.g. "0^"
    orb_idxs = []
    daggers = []
    # properties of multi-body operators, e.g. "0^ 1"
    weights = []
    # herm_term = []
    diag_idxs = []
    off_diag_idxs = []
    # below connect the second type to the first type (used to split single-fermion lists)
    term_split_idxs = []
    constants = []

    term_counter = 0
    single_op_counter = 0
    for term, weight in operators.items():
        if len(term) == 0:
            constants.append(weight)
            continue
        if not all(len(t) == 2 for t in term):  # pragma: no cover
            raise ValueError(f"terms must contain (i, dag) pairs, but received {term}")

        # fill some info about the term
        weights.append(weight)
        is_diag = _is_diag_term(term)
        if is_diag:
            diag_idxs.append(term_counter)
        else:
            off_diag_idxs.append(term_counter)

        # single-fermion operators
        for orb_idx, dagger in term:
            # orb_idxs: holds the hilbert index of the orbital
            orb_idxs.append(orb_idx)
            # daggers: stores whether operator is creator or annihilator
            daggers.append(not bool(dagger))
            single_op_counter += 1

        term_split_idxs.append(single_op_counter)
        term_counter += 1

    orb_idxs = np.array(orb_idxs, dtype=np.intp)
    daggers = np.array(daggers, dtype=bool)
    weights = np.array(weights, dtype=dtype)
    # term_ends = np.array(term_ends, dtype=bool)
    # herm_term = np.array(herm_term, dtype=bool)
    diag_idxs = np.array(diag_idxs, dtype=np.intp)
    off_diag_idxs = np.array(off_diag_idxs, dtype=np.intp)
    term_split_idxs = np.array(term_split_idxs, dtype=np.intp)
    constant = sum(constants)

    return (
        orb_idxs,
        daggers,
        weights,
        diag_idxs,
        off_diag_idxs,
        term_split_idxs,
        constant,
    )


@numba.jit(nopython=True)
def _isclose(a, b, cutoff=1e-6):  # pragma: no cover
    return np.abs(a - b) < cutoff


@numba.jit(nopython=True)
def _is_empty(site):  # pragma: no cover
    return _isclose(site, 0)


@numba.jit(nopython=True)
def _flip(site):  # pragma: no cover
    return 1 - site


@numba.jit(nopython=True)
def _apply_operator(xt, orb_idx, dagger, mel):  # pragma: no cover
    has_xp = True
    empty_site = _is_empty(xt[orb_idx])
    if dagger:
        if not empty_site:
            has_xp = False
        else:
            mel *= (-1) ** np.sum(xt[:orb_idx])  # jordan wigner sign
            xt[orb_idx] = _flip(xt[orb_idx])
    else:
        if empty_site:
            has_xp = False
        else:
            mel *= (-1) ** np.sum(xt[:orb_idx])  # jordan wigner sign
            xt[orb_idx] = _flip(xt[orb_idx])
    if _isclose(mel, 0):
        has_xp = False
    return xt, mel, has_xp
