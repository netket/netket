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

from typing import List, Union, Optional
import re
from collections import defaultdict

import numpy as np
from numba import jit
from jax.tree_util import tree_map
import numbers
import copy
import numba as nb

from netket.utils.types import DType, Array
from netket.operator._discrete_operator import DiscreteOperator
from netket.operator._pauli_strings import _count_of_locations
from netket.hilbert.abstract_hilbert import AbstractHilbert
from netket.utils.numbers import is_scalar

from netket.experimental.hilbert import SpinOrbitalFermions


class FermionOperator2nd(DiscreteOperator):
    r"""
    A fermionic operator in :math:`2^{nd}` quantization.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        terms: Union[List[str], List[List[List[int]]]],
        weights: Optional[List[Union[float, complex]]] = None,
        constant: Union[float, complex] = 0.0,
        dtype: DType = None,
    ):
        r"""
        Constructs a fermion operator given the single terms (set of
        creation/annihilation operators) in second quantization formalism.

        This class can be initialized in the following form:
        `FermionOperator2nd(hilbert, terms, weights ...)`.
        The term contains pairs of `(idx, dagger)`, where `idx ∈ range(hilbert.size)`
        (it identifies an orbital) and `dagger` is a True/False flag determining if the
        operator is a creation or destruction operator.
        A term of the form :math:`\hat{a}_1^\dagger \hat{a}_2` would take the form
        `((1,1), (2,0))`, where (1,1) represents :math:`\hat{a}_1^\dagger` and (2,0)
        represents :math:`\hat{a}_2`.
        To split up per spin, use the creation and annihilation operators to build the
        operator.

        Args:
            hilbert: hilbert of the resulting FermionOperator2nd object
            terms: single term operators (see
                example below)
            weights: corresponding coefficients of the single term operators
                (defaults to a list of 1)
            constant: constant contribution, corresponding to the
                identity operator * constant (default = 0)

        Returns:
            A FermionOperator2nd object.

        Example:
            Constructs the fermionic hamiltonian in :math:`2^{nd}` quantization
            :math:`(0.5-0.5j)*(a_0^\dagger a_1) + (0.5+0.5j)*(a_2^\dagger a_1)`.

            >>> import netket.experimental as nkx
            >>> terms, weights = (((0,1),(1,0)),((2,1),(1,0))), (0.5-0.5j,0.5+0.5j)
            >>> hi = nkx.hilbert.SpinOrbitalFermions(3)
            >>> op = nkx.operator.FermionOperator2nd(hi, terms, weights)
            >>> op
            FermionOperator2nd(hilbert=SpinOrbitalFermions(n_orbitals=3), n_operators=2, dtype=complex128)
            >>> terms = ("0^ 1", "2^ 1")
            >>> op = nkx.operator.FermionOperator2nd(hi, terms, weights)
            >>> op
            FermionOperator2nd(hilbert=SpinOrbitalFermions(n_orbitals=3), n_operators=2, dtype=complex128)
            >>> op.hilbert
            SpinOrbitalFermions(n_orbitals=3)
            >>> op.hilbert.size
            3

        """
        super().__init__(hilbert)

        # bring terms, weights into consistent form, autopromote dtypes if necessary
        _operators, _constant, dtype = _canonicalize_input(
            terms, weights, constant, dtype
        )
        self._dtype = dtype

        # we keep the input, in order to be able to add terms later
        self._operators = _operators
        self._constant = _constant

        self._initialized = False
        self._is_hermitian = None  # set when requested

    def _reset_caches(self):
        """
        Cleans the internal caches built on the operator.
        """
        self._initialized = False
        self._is_hermitian = None

    def _setup(self, force: bool = False):
        """Analyze the operator strings and precompute arrays for get_conn inference"""
        if force or not self._initialized:

            # following lists will be used to compute matrix elements
            # they are filled in _add_term
            out = _pack_internals(self._operators, self._dtype)
            self._orb_idxs, self._daggers, self._weights, self._term_ends = out

            self._initialized = True

    @staticmethod
    def from_openfermion(
        hilbert: AbstractHilbert,
        of_fermion_operator: "openfermion.ops.FermionOperator" = None,  # noqa: F821
        *,
        n_orbitals: Optional[int] = None,
        convert_spin_blocks: bool = False,
    ):
        r"""
        Converts an openfermion FermionOperator into a netket FermionOperator2nd.

        The hilbert first argument can be dropped, see __init__ for details and default
        value.
        Warning: convention of openfermion.hamiltonians is different from ours: instead
        of strong spin components as subsequent hilbert state outputs (i.e. the 1/2 spin
        components of spin-orbit i are stored in locations (2*i, 2*i+1)), we concatenate
        blocks of definite spin (i.e. locations (i, n_orbitals+i)).

        Args:
            hilbert: (optional) hilbert of the resulting FermionOperator2nd object
            of_fermion_operator: openfermion.ops.FermionOperator object
            n_orbitals: (optional) total number of orbitals in the system, default
                None means inferring it from the FermionOperator2nd. Argument is
                ignored when hilbert is given.
            convert_spin_blocks: whether or not we need to convert the FermionOperator
                to our convention. Only works if hilbert is provided and if it has
                spin != 0

        Returns:
            A FermionOperator2nd object.
        """
        from openfermion.ops import FermionOperator

        if hilbert is None:
            raise ValueError(
                "The first argument `from_openfermion` must either be an"
                "openfermion operator or an Hilbert space, followed by"
                "an openfermion operator"
            )

        if not isinstance(hilbert, AbstractHilbert):
            # if first argument is not Hilbert, then shift all arguments by one
            hilbert, of_fermion_operator = None, hilbert

        if not isinstance(of_fermion_operator, FermionOperator):
            raise NotImplementedError()

        if convert_spin_blocks and hilbert is None:
            raise ValueError("if convert_spin_blocks, the hilbert must be specified")

        terms = list(of_fermion_operator.terms.keys())
        weights = list(of_fermion_operator.terms.values())
        terms, weights, constant = _collect_constants(terms, weights)

        if hilbert is not None:
            # no warning, just overwrite
            n_orbitals = hilbert.n_orbitals

            if convert_spin_blocks:
                if not hasattr(hilbert, "spin") or hilbert.spin is None:
                    raise ValueError(
                        f"cannot convert spin blocks for hilbert space {hilbert} without spin"
                    )
                n_spin = hilbert._n_spin_states
                terms = _convert_terms_to_spin_blocks(terms, n_orbitals, n_spin)
        if n_orbitals is None:
            # we always start counting from 0, so we only determine the maximum location
            n_orbitals = _count_of_locations(of_fermion_operator)
        if hilbert is None:
            hilbert = SpinOrbitalFermions(n_orbitals)  # no spin splitup assumed

        return FermionOperator2nd(hilbert, terms, weights=weights, constant=constant)

    def __repr__(self):
        return (
            f"FermionOperator2nd(hilbert={self.hilbert}, "
            f"n_operators={len(self._operators)}, dtype={self.dtype})"
        )

    @property
    def dtype(self) -> DType:
        """The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩."""
        return self._dtype

    def copy(self, *, dtype: Optional[DType] = None):
        if dtype is None:
            dtype = self.dtype
        if not np.can_cast(self.dtype, dtype, casting="same_kind"):
            raise ValueError(f"Cannot cast {self.dtype} to {dtype}")
        op = FermionOperator2nd(
            self.hilbert, [], [], constant=self._constant, dtype=dtype
        )
        op._operators = copy.deepcopy(self._operators)
        return op

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        if _isclose(self._constant, 0):
            return len(self._operators)
        else:
            return len(self._operators) + 1  # constant also can add a term

    @property
    def is_hermitian(self) -> bool:
        """Returns true if this operator is hermitian."""
        if self._is_hermitian is None:  # only compute when needed, is expensive
            terms = list(self._operators.keys())
            weights = list(self._operators.values())
            self._is_hermitian = _check_hermitian(terms, weights)
        return self._is_hermitian

    def operator_string(self):
        """Return a readable string describing all the operator terms"""
        op_string = []
        if not _isclose(self._constant, 0.0):
            op_string.append(f"{self._constant} []")
        for term, weight in self._operators.items():
            s = []
            for idx, dag in term:
                dag_string = "^" if bool(dag) else ""
                s.append(f"{int(idx)}{dag_string}")
            s = " ".join(s)
            s = f"{weight} [{s}]"
            op_string.append(s)
        return " +\n".join(op_string)

    def _get_conn_flattened_closure(self):
        self._setup()
        _max_conn_size = self.max_conn_size
        _orb_idxs = self._orb_idxs
        _daggers = self._daggers
        _weights = self._weights
        _term_ends = self._term_ends
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
                _term_ends,
                _constant,
            )

        return jit(nopython=True)(gccf_fun)

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
            self._constant,
            pad,
        )

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(
        x,
        sections,
        max_conn,
        orb_idxs,
        daggers,
        weights,
        term_ends,
        constant,
        pad=False,
    ):
        x_prime = np.empty((x.shape[0] * max_conn, x.shape[1]), dtype=x.dtype)
        mels = np.zeros((x.shape[0] * max_conn), dtype=weights.dtype)

        def is_empty(site):
            return _isclose(site, 0)

        def flip(site):
            return 1 - site

        # loop over the batch dimension
        n_c = 0
        for b in range(x.shape[0]):

            xb = x[b, :]

            # we can already fill up with default values
            if pad:
                x_prime[b * max_conn : (b + 1) * max_conn, :] = np.copy(xb)

            if not _isclose(constant, 0.0):
                x_prime[n_c, :] = np.copy(xb)
                mels[n_c] = constant
                n_c += 1

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

    def _op__imatmul__(self, other):
        if not isinstance(other, FermionOperator2nd):
            return NotImplementedError
        if not self.hilbert == other.hilbert:
            raise ValueError(
                "Can only multiply identical hilbert spaces (got A@B, "
                f"A={self.hilbert}, B={other.hilbert})"
            )
        if not np.can_cast(_dtype(other), self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot multiply inplace operator with dtype {type(other)} "
                f"to operator with dtype {self.dtype}"
            )

        terms = []
        weights = []
        for t, w in self._operators.items():
            for to, wo in other._operators.items():
                terms.append(tuple(t) + tuple(to))
                weights.append(w * wo)
        if not _isclose(other._constant, 0.0):
            for t, w in self._operators.items():
                terms.append(tuple(t))
                weights.append(w * other._constant)
        if not _isclose(self._constant, 0.0):
            for t, w in other._operators.items():
                terms.append(tuple(t))
                weights.append(w * self._constant)
        constant = self._constant * other._constant

        self._operators = dict(zip(terms, weights))
        self._constant = constant
        self._reset_caches()

        return self

    def _op__matmul__(self, other):
        if not isinstance(other, FermionOperator2nd):
            return NotImplementedError
        dtype = np.promote_types(self.dtype, other.dtype)
        op = self.copy(dtype=dtype)
        return op._op__imatmul__(other)

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        dtype = np.promote_types(self.dtype, _dtype(other))
        op = self.copy(dtype=dtype)
        return op.__iadd__(other)

    def __iadd__(self, other):
        if is_scalar(other):
            if not _isclose(other, 0.0):
                self._constant += other
            return self
        if not isinstance(other, FermionOperator2nd):
            raise NotImplementedError
        if not self.hilbert == other.hilbert:
            raise ValueError(
                f"Can only add identical hilbert spaces (got A+B, A={self.hilbert}, "
                "B={other.hilbert})"
            )
        if not np.can_cast(_dtype(other), self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot add inplace operator with dtype {type(other)} "
                f"to operator with dtype {self.dtype}"
            )
        for t, w in other._operators.items():
            if t in self._operators.keys():
                self._operators[t] += w
            else:
                self._operators[t] = w
        self._constant += other._constant
        self._reset_caches()
        return self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self.__mul__(np.array(-1, dtype=self.dtype))

    def __rmul__(self, scalar):
        return self * scalar

    def __imul__(self, scalar):
        if not is_scalar(scalar):
            # we will overload this as matrix multiplication
            self._op__imatmul__(scalar)
        if not np.can_cast(_dtype(scalar), self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot multiply inplace scalar with dtype {type(scalar)} "
                f"to operator with dtype {self.dtype}"
            )
        scalar = np.array(scalar, dtype=self.dtype).item()
        self._operators = tree_map(lambda x: x * scalar, self._operators)
        self._constant *= scalar
        self._reset_caches()
        return self

    def __mul__(self, scalar):
        if not is_scalar(scalar):
            # we will overload this as matrix multiplication
            return self._op__matmul__(scalar)
        dtype = np.promote_types(self.dtype, _dtype(scalar))
        op = self.copy(dtype=dtype)
        return op.__imul__(scalar)

    def conjugate(self, *, concrete=False):
        r"""Returns the complex conjugate of this operator."""

        terms = list(self._operators.keys())
        weights = list(self._operators.values())
        terms, weights = _herm_conj(terms, weights)  # changes also the terms
        terms = _make_tuple_tree(terms)

        new = FermionOperator2nd(
            self.hilbert,
            [],
            [],
            constant=np.conjugate(self._constant),
            dtype=self.dtype,
        )
        new._operators = dict(zip(terms, weights))
        return new


def _convert_terms_to_spin_blocks(terms, n_orbitals: int, n_spin_components: int):
    """
    See explanation in from_openfermion in conversion between conventions of netket
    and openfermion.

    Args:
        terms: the operator terms in tuple tree format
        n_orbitals: number of orbitals
        n_spin_components: number of spin components (2*spin+1)

    Returns:
        new terms tree
    """

    if n_spin_components == 1:
        return terms

    def _convert_loc(of_loc):
        orb_idx = of_loc // n_spin_components
        spin_idx = of_loc % n_spin_components
        return orb_idx + n_orbitals * spin_idx

    def _convert_term(term):
        return tuple([(_convert_loc(t[0]), t[1]) for t in term])

    return tuple(list(map(_convert_term, terms)))


def _collect_constants(terms, weights):
    """
    Openfermion has the convention to store constants as empty terms
    Returns new terms and weights list, and the collected constants
    """
    new_terms = []
    new_weights = []
    constant = 0.0
    for t, w in zip(terms, weights):
        if len(t) == 0:
            constant += w
        else:
            new_terms.append(t)
            new_weights.append(w)
    return new_terms, new_weights, constant


def _canonicalize_input(terms, weights, constant, dtype):
    """The canonical form is a tree tuple with a tuple pair of integers at the
    lowest level"""
    if isinstance(terms, str):
        terms = (terms,)

    if len(terms) > 0:
        terms = _parse_term_tree(terms)

    if weights is None:
        weights = [1.0] * len(terms)

    # promote dtype iwth constant
    if dtype is None:
        constant_dtype = np.array(constant).dtype
        weights_dtype = np.array(weights).dtype
        dtype = np.promote_types(constant_dtype, weights_dtype)

    weights = np.array(weights, dtype=dtype).tolist()
    constant = np.array(constant, dtype=dtype).item()

    if not len(weights) == len(terms):
        raise ValueError(
            f"length of weights should be equal, but received {len(weights)} and {len(terms)}"
        )

    _check_tree_structure(terms)

    operators = dict(zip(terms, weights))

    return operators, constant, dtype


def _parse_term_tree(terms):
    """convert the terms tree into a canonical form of tuple tree of depth 3"""

    def _parse_branch(t):
        if isinstance(t, str):
            return _parse_string(t)
        elif hasattr(t, "__len__"):
            return tuple([_parse_branch(b) for b in t])
        else:
            return int(t)

    return _parse_branch(terms)


def _parse_string(s):
    """parse strings such as '1^ 2' into a term form ((1, 1), (2, 0))"""
    s = s.strip()
    if s == "":
        return ()
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
    return tuple(processed_terms)


def _check_hermitian(
    terms: List[List[List[int]]], weights: Union[float, complex] = 1.0
):
    """
    Check whether a set of terms and weights for a hermitian operator
    The terms are ordered into a canonical form with daggers and high orbital numbers to the left.
    After conjugation, the result is again reordered into canonical form.
    The result of both ordered lists of terms and weights are compared to be the same
    """
    # save in a dictionary the normal ordered terms and weights
    normal_ordered = _normal_ordering(terms, weights)
    dict_normal = defaultdict(complex)
    for term, weight in zip(*normal_ordered):
        dict_normal[tuple(term)] += weight

    # take the hermitian conjugate of the terms
    hc = _herm_conj(terms, weights)
    # normal order the h.c. terms
    hc_normal_ordered = _normal_ordering(*hc)

    # save in a dictionary the normal ordered h.c. terms and weights
    dict_hc_normal = defaultdict(complex)
    for term_hc, weight_hc in zip(*hc_normal_ordered):
        dict_hc_normal[tuple(term_hc)] += weight_hc

    # check if hermitian by comparing the dictionaries
    dict_normal = dict(dict_normal)
    dict_hc_normal = dict(dict_hc_normal)
    is_hermitian = dict_normal == dict_hc_normal
    return is_hermitian


def _order_fun(term: List[List[int]], weight: Union[float, complex] = 1.0):
    """
    Return a normal ordered single term of the fermion operator.
    Normal ordering corresponds to placing the operator acting on the
    highest index on the left and lowest index on the right. In addition,
    the creation operators are placed on the left and annihilation on the right.
    In this ordering, we make sure to account for the anti-commutation of operators.
    """

    parity = -1
    term = copy.deepcopy(list(term))
    weight = copy.copy(weight)
    ordered_terms = []
    ordered_weights = []
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

                    # ad the processed term
                    o, w = _order_fun(tuple(new_term), parity * weight)
                    ordered_terms += o
                    ordered_weights += w

            # if we have two creation or two annihilation operators
            elif right_term[1] == left_term[1]:

                # If same two Fermionic operators are repeated,
                # evaluate to zero.
                if parity == -1 and right_term[0] == left_term[0]:
                    return ordered_terms, ordered_weights

                # swap if same type but order is not correct
                elif right_term[0] > left_term[0]:
                    term[j - 1] = right_term
                    term[j] = left_term
                    weight *= parity

    ordered_terms.append(term)
    ordered_weights.append(weight)
    return ordered_terms, ordered_weights


def _normal_ordering(
    terms: List[List[List[int]]], weights: List[Union[float, complex]] = 1
):
    """
    Returns the normal ordered terms and weights of the fermion operator.
    We use the following normal ordering convention: we order the terms with
    the highest index of the operator on the left and the lowest index on the right. In addition,
    creation operators are placed on the left and annihilation operators on the right.
    """
    ordered_terms = []
    ordered_weights = []
    # loop over all the terms and weights and order each single term with corresponding weight
    for term, weight in zip(terms, weights):
        ordered = _order_fun(term, weight)
        ordered_terms += ordered[0]
        ordered_weights += ordered[1]
    ordered_terms = _make_tuple_tree(ordered_terms)
    return ordered_terms, ordered_weights


def _make_tuple_tree(terms):
    """make tuples, so terms are hashable"""

    def _make_tuple(branch):
        if hasattr(branch, "__len__"):
            return tuple([_make_tuple(t) for t in branch])
        else:
            return int(branch)

    return _make_tuple(terms)


def _herm_conj(terms: List[List[List[int]]], weights: List[Union[float, complex]] = 1):
    """Returns the hermitian conjugate of the terms and weights."""
    conj_term = []
    conj_weight = []
    # loop over all terms and weights and get the hermitian conjugate
    for term, weight in zip(terms, weights):
        conj_term.append([(op, 1 - int(dag)) for (op, dag) in reversed(term)])
        conj_weight.append(np.conjugate(weight))
    return conj_term, conj_weight


def _check_tree_structure(terms):
    """Check whether the terms structure is depth 3 everywhere and contains pairs of (idx, dagger) everywhere"""

    def _descend(tree, current_depth, depths, pairs):
        if current_depth == 2 and hasattr(tree, "__len__"):
            pairs.append(len(tree) == 2)
        if hasattr(tree, "__len__"):
            for branch in tree:
                _descend(branch, current_depth + 1, depths, pairs)
        else:
            depths.append(current_depth)

    depths = []
    pairs = []
    _descend(terms, 0, depths, pairs)
    if not np.all(np.array(depths) == 3):
        raise ValueError(f"terms is not a depth 3 tree, found depths {depths}")
    if not np.all(pairs):
        raise ValueError("terms should be provided in (i, dag) pairs")


def _pack_internals(operators, dtype):
    """
    Create the internal structures to compute the matrix elements
    Processes and adds a single term such that we can compute its matrix elements, in tuple format ((1,1), (2,0))
    """
    orb_idxs = []
    daggers = []
    term_ends = []
    weights = []

    for term, weight in operators.items():
        if len(term) == 0:
            raise ValueError("terms cannot be size 0")
        if not all(len(t) == 2 for t in term):
            raise ValueError(f"terms must contain (i, dag) pairs, but received {term}")
        for orb_idx, dagger in reversed(term):
            # orb_idxs: holds the hilbert index of the orbital
            orb_idxs.append(orb_idx)
            # daggers: stores whether operator is creator or annihilator
            daggers.append(bool(dagger))
            # weights: stores the weights corresponding to this term
            weights.append(weight)
            # term_ends keeps a flag that determines whether we've reached the end of this term and will start a new one after
            term_ends.append(False)
        term_ends[-1] = True

    orb_idxs = np.array(orb_idxs, dtype=np.intp)
    daggers = np.array(daggers, dtype=bool)
    weights = np.array(weights, dtype=dtype)
    term_ends = np.array(term_ends, dtype=bool)

    return orb_idxs, daggers, weights, term_ends


def _dtype(obj: Union[numbers.Number, Array, "FermionOperator2nd"]) -> DType:
    """
    Returns the dtype of the input object
    """
    if isinstance(obj, numbers.Number):
        return type(obj)
    elif isinstance(obj, DiscreteOperator):
        return obj.dtype
    elif isinstance(obj, np.ndarray):
        return obj.dtype
    else:
        raise TypeError(f"cannot deduce dtype of object type {type(obj)}: {obj}")


@nb.jit(nopython=True)
def _isclose(a, b, cutoff=1e-6):
    return np.abs(a - b) < cutoff
