import re
from collections import defaultdict
import numpy as np
from numbers import Number
import copy

from netket.jax import canonicalize_dtypes
from netket.utils.types import DType, PyTree

OperatorTuple = tuple[int, int]
r""" Creation and annihilation operators at mode i are encoded as
:math:`\hat{a}_i^\dagger`: (i, 1)
:math:`\hat{a}`: (i, 0)
"""

OperatorTerm = tuple[OperatorTuple, ...]
r""" A term of the form :math:`\hat{a}_1^\dagger \hat{a}_2` would take the form
`((1,1), (2,0))`, where (1,1) represents :math:`\hat{a}_1^\dagger` and (2,0)
represents :math:`\hat{a}_2`."""

OperatorTermsList = list[OperatorTerm]
""" A list of operators that would e.g. describe a Hamiltonian """

OperatorWeightsList = list[Number]
""" A list of weights of corresponding terms """

OperatorDict = dict[OperatorTerm, Number]
""" A dict containing OperatorTerm as key and weights as the values """


def _normal_order_term(
    term: OperatorTerm, weight: Number = 1.0
) -> tuple[OperatorTermsList, OperatorWeightsList]:
    """
    Return a normal ordered single term of the fermion operator.
    Normal ordering corresponds to placing creating operators on the left
    and annihilation on the right.
    Then, it places the highest index on the left and lowest index on the right
    In this ordering, we make sure to account for the anti-commutation of operators.
    """

    parity = -1
    term = copy.deepcopy(list(term))
    weight = copy.copy(weight)

    if len(term) == 0:  # a constant
        return [term], [weight]
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
                    o, w = _normal_order_term(tuple(new_term), parity * weight)
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
    terms: OperatorTermsList, weights: OperatorWeightsList = 1.0
) -> tuple[OperatorTermsList, OperatorWeightsList]:
    """
    Returns the normal ordered terms and weights of the fermion operator.
    """
    ordered_terms = []
    ordered_weights = []
    # loop over all the terms and weights and order each single term with corresponding weight
    for term, weight in zip(terms, weights):
        ordered = _normal_order_term(term, weight)
        ordered_terms += ordered[0]
        ordered_weights += ordered[1]
    ordered_terms = _make_tuple_tree(ordered_terms)
    return ordered_terms, ordered_weights


def _pair_order_term(
    term: OperatorTerm, weight: Number = 1.0
) -> tuple[OperatorTermsList, OperatorWeightsList]:
    """
    Return a pair ordered single term of the fermion operator.
    Pair ordering corresponds to placing first the highest indices on the right,
    and then making sure the creation operators are on the left and annihilation on the right.
    In this ordering, we make sure to account for the anti-commutation of operators.
    """

    parity = -1
    term = copy.deepcopy(list(term))
    weight = copy.copy(weight)
    if len(term) == 0:
        return [term], [weight]
    ordered_terms = []
    ordered_weights = []
    # the arguments given to this function will be transformed in a normal ordered way
    # loop through all the operators in the single term from left to right and order them
    # by swapping the term operators (and transform the weights by multiplying with the parity factors)
    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            right_term = term[j]
            left_term = term[j - 1]

            ## exchange operators if biggest is on the right (need commutataion relations)
            # exchange operators if biggest is on the left (need commutataion relations)
            if right_term[0] < left_term[0]:  # not the same, always switch
                # fulfil anti commutation relations
                term[j - 1] = right_term
                term[j] = left_term
                weight *= parity

            # exchange operators if creation operator is on the right and annihilation on the left
            elif right_term[0] == left_term[0]:
                if right_term[1] and not left_term[1]:
                    term[j - 1] = right_term
                    term[j] = left_term
                    weight *= parity

                    # if same indices switch order (creation on the left), remember a a^ = 1 + a^ a
                    # if right_term[0] == left_term[0]:
                    new_term = term[: (j - 1)] + term[(j + 1) :]

                    # ad the processed term
                    o, w = _pair_order_term(tuple(new_term), parity * weight)
                    ordered_terms += o
                    ordered_weights += w

                # if we have two creation or two annihilation operators
                elif right_term[1] == left_term[1]:
                    # If same two Fermionic operators are repeated,
                    # evaluate to zero.
                    # if parity == -1 and right_term[0] == left_term[0]:
                    return ordered_terms, ordered_weights

    ordered_terms.append(term)
    ordered_weights.append(weight)
    return ordered_terms, ordered_weights


def _pair_ordering(
    terms: OperatorTermsList, weights: OperatorWeightsList = 1.0
) -> tuple[OperatorTermsList, OperatorWeightsList]:
    """
    Returns the pair ordered terms and weights of the fermion operator.
    """
    ordered_terms = []
    ordered_weights = []
    # loop over all the terms and weights and order each single term with corresponding weight
    for term, weight in zip(terms, weights):
        ordered = _pair_order_term(term, weight)
        ordered_terms += ordered[0]
        ordered_weights += ordered[1]
    ordered_terms = _make_tuple_tree(ordered_terms)
    return ordered_terms, ordered_weights


def _check_hermitian(
    terms: OperatorTermsList, weights: OperatorWeightsList, cutoff: float
) -> bool:
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

    # compare dict up to a tolerance (1e-10)
    is_hermitian = _dict_compare(dict_normal, dict_hc_normal, cutoff)

    return is_hermitian


def _herm_conj(
    terms: OperatorTermsList, weights: OperatorWeightsList = 1
) -> tuple[OperatorTermsList, OperatorWeightsList]:
    """Returns the hermitian conjugate of the terms and weights."""
    conj_term = transpose_terms(terms)
    conj_weight = [np.conjugate(weight) for weight in weights]
    return conj_term, conj_weight


def transpose_terms(terms: OperatorTermsList) -> OperatorTermsList:
    """Returns the transpose of the terms (creation/destruction operators).
    The transpose is equivalent to the hermitian transpose because they are
    real.
    """
    conj_term = []
    for term in terms:
        conj_term.append([(op, 1 - int(dag)) for (op, dag) in reversed(term)])
    return conj_term


def transpose_term(term: OperatorTerm) -> OperatorTerm:
    """Returns the transpose of a single term (creation/destruction operators).
    The transpose is equivalent to the hermitian transpose because they are
    real.
    """
    # equivalent to transpose_terms
    return tuple((op, 1 - int(dag)) for (op, dag) in reversed(term))


def _convert_terms_to_spin_blocks(
    terms: OperatorTermsList, n_orbitals: int, n_spin_components: int
) -> OperatorTermsList:
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


def _canonicalize_input(
    terms: OperatorTermsList,
    weights: OperatorWeightsList,
    dtype: DType | None,
    cutoff: float,
    constant: Number = 0,
) -> tuple[OperatorDict, Number, DType]:
    r"""
    The canonical form is a tree tuple with a tuple pair of integers at the
    lowest level

    A term of the form :math:`\hat{a}_1^\dagger \hat{a}_2` would take the form
        `((1,1), (2,0))`, where (1,1) represents :math:`\hat{a}_1^\dagger` and (2,0)
        represents :math:`\hat{a}_2`.
    """
    if terms is None:
        terms = []

    if isinstance(terms, str):
        terms = (terms,)

    if len(terms) > 0:
        terms = _parse_term_tree(terms)

    if weights is None:
        weights = [1.0] * len(terms)

    weights = list(weights)
    # convert a constant to a diagonal operator
    if not np.isclose(constant, 0.0, atol=cutoff):
        terms = [()] + list(terms)
        weights = [constant] + weights

    dtype = canonicalize_dtypes(float, *weights, constant, dtype=dtype)

    weights = np.array(weights, dtype=dtype).tolist()

    if not len(weights) == len(terms):
        raise ValueError(
            f"length of weights should be equal, but received {len(weights)} and {len(terms)}"
        )

    _check_tree_structure(terms)

    # add the weights of terms that occur multiple times
    operators = zero_defaultdict(dtype)
    for t, w in zip(terms, weights):
        operators[t] += w
    operators = _remove_dict_zeros(dict(operators), cutoff)

    return operators, dtype


def _verify_input(hilbert, operators, raise_error=True) -> bool:
    """Check whether all input is valid"""
    terms = list(operators.keys())

    def _check_op(fop):
        v1 = 0 <= fop[0] < hilbert.size
        if not v1:
            if raise_error:
                raise ValueError(
                    f"Found invalid orbital index {fop[0]} for hilbert space {hilbert} of size {hilbert.size}"
                )
            return False
        v2 = fop[1] in (0, 1)
        if not v2:
            if raise_error:
                raise ValueError(
                    f"Found invalid character {fop[1]} for dagger, which should be 0 (no dagger) or 1 (dagger)."
                )
            return False
        return True

    def _check_term(term):
        return all(_check_op(t) for t in term)

    return all(_check_term(term) for term in terms)


def _remove_dict_zeros(d: dict, cutoff: float) -> dict:
    """Remove redundant zero values from a dictionary"""
    return {k: v for k, v in d.items() if np.abs(v) > cutoff}


def _parse_term_tree(terms: OperatorTermsList) -> OperatorTermsList:
    """Convert the terms tree into a canonical form of tuple tree of depth 3"""

    def _parse_branch(t):
        if isinstance(t, str):
            return _parse_string(t)
        elif hasattr(t, "__len__"):
            return tuple([_parse_branch(b) for b in t])
        else:
            return int(t)

    return _parse_branch(terms)


def _parse_string(s: str) -> OperatorTerm:
    """Parse strings such as '1^ 2' into a term form ((1, 1), (2, 0))"""
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


def _dict_compare(d1: dict, d2: dict, cutoff: float) -> bool:
    """
    Compare two dicts and return True if their keys and values
    are all the same (up to some tolerance)
    """
    d1 = _remove_dict_zeros(d1, cutoff)
    d2 = _remove_dict_zeros(d2, cutoff)
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    if d1_keys != d2_keys:
        return False
    # We checked that d1 and d2 have the same keys. Now check the values.
    return all(np.isclose(d1[o], d2[o], atol=cutoff) for o in d1_keys)


def _make_tuple_tree(terms: PyTree) -> PyTree:
    """Make tuples, so terms are hashable.

    Input could be e.g. a pytree of lists of lists,
    which we convert to tuples of tuples.
    """

    def _make_tuple(branch):
        if hasattr(branch, "__len__"):
            return tuple([_make_tuple(t) for t in branch])
        else:
            return int(branch)

    return _make_tuple(terms)


def _check_tree_structure(terms: OperatorTermsList) -> OperatorTermsList:
    """
    Check whether the terms structure is depth 3 everywhere
    and contains pairs of (idx, dagger) everywhere
    """

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
        raise ValueError(
            "terms should be provided in (i, dag) pairs or empty for a constant"
        )


def _is_diag_term(term: OperatorTerm) -> bool:
    """
    Check whether this term changes the sample or not
    """

    def _init_empty_arr():
        return [
            0,
            0,
        ]  # first one counts number of daggers, second number of non-daggers

    if len(term) == 0:
        return True  # constant

    ops = defaultdict(_init_empty_arr)
    for orb_idx, dagger in term:
        ops[orb_idx][int(dagger)] += 1
    return all((x[0] == x[1]) for x in ops.values())


def zero_defaultdict(dtype: DType) -> defaultdict:
    """
    Temporary function to make sure we get a good initializer for each dtype
    """

    def _dtype_init():
        return np.array(0, dtype=dtype).item()

    return defaultdict(_dtype_init)
