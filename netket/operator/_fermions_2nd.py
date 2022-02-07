from typing import List, Union
from netket.utils.types import DType

import numpy as np
from numba import jit

from netket.hilbert import AbstractHilbert, SpinOrbitalFermions

from netket.operator._discrete_operator import DiscreteOperator
from netket.operator._pauli_strings import _count_of_locations

import re


def _parse_string(s):
    s = s.strip()
    s = re.sub(" +", " ", s)
    terms = s.split(" ")
    processed_terms = []
    for term in terms:
        if term[-1] == "^":
            dagger = True
            term = term[:-1]
        else:
            dagger = False
        orb_nr = int(term)
        processed_terms.append((orb_nr, int(dagger)))
    processed_terms = tuple(processed_terms)
    return processed_terms


class FermionOperator2nd(DiscreteOperator):
    """Constructs a fermion operator given the single terms (set of creation/annihilation operators) in second quantization formalism."""

    def __init__(
        self,
        hilbert: AbstractHilbert,
        terms: Union[List[List[str]], List[List[List[int]]]],
        weights: List[Union[float, complex]] = None,
        dtype: DType = complex,
    ):

        r"""
        This class can be initialized in the following form: ``FermionOperator2nd(hilbert, terms, weights ...)``.
        Args:
            hilbert (required): hilbert of the resulting FermionOperator2nd object
            terms (list(list(list(int)))): single term operators
            weights (list(union(float,complex))): corresponding coefficients of the single term operators
        Returns:
            A FermionOperator2nd object.

        Example:
            Constructs a new ``FermionOperator2nd`` operator (0.5-0.5j)*(a_0^dagger a_1) + (0.5+0.5j)*(a_2^dagger a_1)  with the construction scheme.
            >>> import netket as nk
            >>> terms,weights = (((0,1),(1,0)),((2,1),(1,0))), (0.5-0.5j,0.5+0.5j)
            >>> hi = nk.hilbert.SpinOrbitalFermions(3)
            >>> op = nk.operator.FermionOperator2nd(hi, terms, weights)
            >>> op
            FermionOperator2nd(hilbert=Fock(n_max=1, N=3), n_terms=2)
            >>> terms = ("0^ 1", "2^ 1")
            >>> op = nk.operator.FermionOperator2nd(hi, terms, weights)
            >>> op
            FermionOperator2nd(hilbert=Fock(n_max=1, N=3), n_terms=2)
            >>> op.hilbert
            Fock(n_max=1, N=3)
            >>> op.hilbert.size
            3
        """
        super().__init__(hilbert)
        self._dtype = dtype
        self._orb_idxs = []
        self._daggers = []
        self._term_ends = []
        self._weights = []
        self._n_terms = 0
        # we keep the input, in order to be able to add terms later
        self._orig_terms = []
        self._orig_weights = []

        if isinstance(terms, str):
            terms = (terms,)

        if len(terms) > 0 and isinstance(terms[0], str):
            terms = list(map(_parse_string, terms))

        if weights is None:
            weights = [1.0] * len(terms)

        if not len(weights) == len(terms):
            raise ValueError("length of weights should be equal")

        for term, weight in zip(terms, weights):
            self.add_term(term, weight=weight)

        self._initialized = False
        self._is_hermitian = _check_hermitian(self._orig_terms, self._orig_weights)

    def add_term(self, term, weight=1.0):
        if isinstance(term, str):
            term = _parse_string(term)

        self._orig_terms.append(term)
        self._orig_weights.append(weight)

        for orb_idx, dagger in reversed(term):
            self._orb_idxs.append(orb_idx)
            self._daggers.append(bool(dagger))
            self._weights.append(weight)
            self._term_ends.append(False)
        self._term_ends[-1] = True
        self._n_terms += 1

    def _setup(self, force=False):
        """Analyze the operator strings and precompute arrays for get_conn inference"""
        if force or not self._initialized:
            self._orb_idxs = np.array(self._orb_idxs, dtype=np.intp)
            self._daggers = np.array(self._daggers, dtype=bool)
            self._weights = np.array(self._weights, dtype=self.dtype)
            self._term_ends = np.array(self._term_ends, dtype=bool)
            self._initialized = True

    @staticmethod
    def from_openfermion(
        hilbert: AbstractHilbert,
        of_fermion_operator: "openfermion.ops.FermionOperator" = None,  # noqa: F821
        *,
        n_orbitals: int = None,
    ):
        r"""
        Converts an openfermion FermionOperator into a netket FermionOperator2nd.
        The hilbert first argument can be dropped, see __init__ for details and default value
        Args:
            hilbert (optional): hilbert of the resulting FermionOperator2nd object
            of_fermion_operator (required): openfermion.ops.FermionOperator object
            n_orbitals (int): total number of orbitals in the system, default None means inferring it from the FermionOperator2nd. Argument is ignored when hilbert is given.
        Returns:
            A FermionOperator2nd object.
        """
        from openfermion.ops import FermionOperator

        if hilbert is None:
            raise ValueError("None-valued hilbert passed.")

        if not isinstance(hilbert, AbstractHilbert):
            # if first argument is not Hilbert, then shift all arguments by one
            hilbert, of_fermion_operator = None, hilbert
        if not isinstance(of_fermion_operator, FermionOperator):
            raise NotImplementedError()

        if hilbert is not None:
            # no warning, just overwrite
            n_orbitals = hilbert.size
        if n_orbitals is None:
            # we always start counting from 0, so we only determine the maximum location
            n_orbitals = _count_of_locations(of_fermion_operator)
        if hilbert is None:
            hilbert = SpinOrbitalFermions(n_orbitals)

        terms = list(of_fermion_operator.terms.keys())
        weights = list(of_fermion_operator.terms.values())
        return FermionOperator2nd(hilbert, terms, weights=weights)

    def __repr__(self):
        return "FermionOperator2nd(hilbert={}, n_terms={})".format(
            self.hilbert, self._n_terms
        )

    @property
    def dtype(self) -> DType:
        """The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩."""
        return self._dtype

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return self._n_terms

    @property
    def is_hermitian(self) -> bool:
        """Returns true if this operator is hermitian."""
        return self._is_hermitian

    def _get_conn_flattened_closure(self):
        self._setup()
        _max_conn_size = self.max_conn_size
        _orb_idxs = self._orb_idxs
        _daggers = self._daggers
        _weights = self._weights
        _term_ends = self._term_ends
        fun = self._flattened_kernel

        def gccf_fun(x, sections):
            return fun(
                x, sections, _max_conn_size, _orb_idxs, _daggers, _weights, _term_ends
            )

        return jit(nopython=True)(gccf_fun)

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
            self.max_conn_size,
            self._orb_idxs,
            self._daggers,
            self._weights,
            self._term_ends,
            pad,
        )

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(
        x, sections, max_conn, orb_idxs, daggers, weights, term_ends, pad=False
    ):
        x_prime = np.empty((x.shape[0] * max_conn, x.shape[1]), dtype=x.dtype)
        mels = np.zeros((x.shape[0] * max_conn), dtype=weights.dtype)

        def is_empty(site):
            return site == 0.0

        def flip(site):
            return 1 - site

        # loop over the batch dimension
        n_c = 0
        for b in range(x.shape[0]):

            xb = x[b, :]

            # we can already fill up with default values
            if pad:
                x_prime[b * max_conn : (b + 1) * max_conn, :] = np.copy(xb)

            new_term = True
            # loop over all terms and sum where necessary
            for orb_idx, dagger, weight, term_end in zip(
                orb_idxs, daggers, weights, term_ends
            ):
                if new_term:
                    new_term = False
                    xt = np.copy(xb)
                    mel = weight
                    has_xp = True

                if has_xp:  # previous term items made a zero, so skip

                    empty_site = is_empty(xt[orb_idx])

                    if dagger:
                        if not empty_site:
                            has_xp = False
                        else:
                            mel *= (-1) ** np.sum(xt[:orb_idx])  # jordan wigner sign
                            xt[orb_idx] = flip(xt[orb_idx])
                    else:
                        if empty_site:
                            has_xp = False
                        else:
                            mel *= (-1) ** np.sum(xt[:orb_idx])  # jordan wigner sign
                            xt[orb_idx] = flip(xt[orb_idx])

                # if this is the end of the term, we collect things
                if term_end:

                    if has_xp:
                        x_prime[n_c, :] = xt
                        mels[n_c] = mel
                        n_c += 1

                    new_term = True

            if pad:
                n_c = (b + 1) * max_conn

            sections[b] = n_c

        if pad:
            return x_prime, mels
        else:
            return x_prime[:n_c], mels[:n_c]


def _check_hermitian(
    terms: List[List[List[int]]], weights: Union[float, complex] = 1.0
):
    """Check whether a set of terms and weights for a hermitian operator"""

    # save in a dictionary the normal ordered terms and weights
    normal_ordered = _normal_ordering(terms, weights)

    dict_normal = {}
    for term, weight in zip(*normal_ordered):
        dict_normal[tuple(term)] = weight

    # take the hermitian conjugate of the terms
    hc = _herm_conj(terms, weights)

    # normal order the h.c. terms
    hc_normal_ordered = _normal_ordering(*hc)

    # save in a dictionary the normal ordered h.c. terms and weights
    dict_hc_normal = {}
    for term_hc, weight_hc in zip(*hc_normal_ordered):
        dict_hc_normal[tuple(term_hc)] = weight_hc

    # check if hermitian by comparing the dictionaries
    is_hermitian = dict_normal == dict_hc_normal
    return is_hermitian


def _order_fun(term: List[List[int]], weight: Union[float, complex] = 1.0):
    """Return a normal ordered single term of the fermion operator.
    Normal ordering corresponds to placing the operator acting on the
    highest index on the left and lowest index on the right. In addition,
    the creation operators are placed on the left and annihilation on the right.
    In this ordering, we make sure to account for the anti-commutation of operators.
    """

    parity = -1
    term = list(term)
    # the arguments given to this function will be transformed in a normal ordered way
    # loop through all the operators in the single term from left to right and order them
    # by swapping the term operators (and transform the weights by multiplying with the parity factors)
    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            right_term = term[j]
            left_term = term[j - 1]

            # exchange operators if creation operator is on the right and annihilation on the left
            if right_term[1] and not left_term[1]:
                term[j - 1] = right_term
                term[j] = left_term
                weight *= parity

                # if same indices switch order (creation on the left), remember a a^ = 1 + a^ a
                if right_term[0] == left_term[0]:
                    new_term = term[: (j - 1)] + term[(j + 1) :]
                    weight *= parity
                    term = new_term

            # if we have two creation or two annihilation operators
            elif right_term[1] == left_term[1]:

                # If same two Fermionic operators are repeated,
                # evaluate to zero.
                if parity == -1 and right_term[0] == left_term[0]:
                    return None, None  # return None if the weight is zero

                # swap if same type but order is not correct
                elif right_term[0] > left_term[0]:
                    term[j - 1] = right_term
                    term[j] = left_term
                    weight *= parity
    return term, weight


def _normal_ordering(
    terms: List[List[List[int]]], weights: List[Union[float, complex]] = 1
):
    """Returns the normal ordered terms and weights of the fermion operator.
    We use the following normal ordering convention: we order the terms with
    the highest index of the operator on the left and the lowest index on the right. In addition,
    creation operators are placed on the left and annihilation operators on the right."""
    ordered_terms = []
    ordered_weights = []
    # loop over all the terms and weights and order each single term with corresponding weight
    for term, weight in zip(terms, weights):
        ordered = _order_fun(term, weight)
        ordered_terms.append(ordered[0])
        ordered_weights.append(ordered[1])
    return ordered_terms, ordered_weights


def _herm_conj(terms: List[List[List[int]]], weights: List[Union[float, complex]] = 1):
    """Returns the hermitian conjugate of the terms and weights."""
    conj_term = []
    conj_weight = []
    # loop over all terms and weights and get the hermitian conjugate
    for term, weight in zip(terms, weights):
        conj_term.append([(op, 1 - action) for (op, action) in reversed(term)])
        conj_weight.append(weight.conjugate())
    return conj_term, conj_weight
