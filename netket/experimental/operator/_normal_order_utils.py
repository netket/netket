# utilities to bring the intermediate internal representation into normal order

import numpy as np
from netket.utils.types import Array


# these types contain the same information as the types in netket.operator._fermion2nd.utils
# but use arrays to store the information

OperatorArrayTerms = tuple[Array, Array, Array]
r"""
A sum of n_terms fermionic operators with fixed n_operators number of creation/annihilation operators
A tuple (sites, daggers, weights) where
    sites: An integer array of size n_terms x n_operators containing the indices i
    daggers: An boolean array of size n_terms x n_operators specifying if the operator is creation/destruction
    weights: An array of size n_terms containing the weight of each term

Example:
The operator :math:`1.0 \hat{c}_1^\dagger \hat{c}_2 + 2.0 \hat{c}_3 \hat{c}_1^\dagger` is represented by
:code:`sites = [[1,2],[3,1]], daggers = [[1,0],[0,1]], weights = [1.0, 2.0]``
"""

OperatorArrayDict = dict[int, OperatorArrayTerms]
r"""
A dictionary containing OperatorArrayTerms of different lengths, where the key specifies the lenght,
representing a generic fermionic operator in second quantization.
"""

SpinOperatorArrayTerms = tuple[Array, Array, Array, Array]
r"""
A sum of n_terms fermionic operators with fixed n_operators number of creation/annihilation operators in n_spin_subsectors different spin secors
A tuple(sites, sectors, daggers, weights) where
    sites: An integer array of size n_terms x n_operators containing the indices i
    sectors: An integer array of size n_terms x n_operators containing the spin sector in 0,...,n_spin_subsectors-1
    daggers: An boolean array of size n_terms x n_operators specifying if the operator is creation/destruction
    weights: An array of size n_terms containing the weight of each term

Example:
The operator :math:`1.0 \hat{c}_{1,\downarrow}^\dagger \hat{c}_{2,\downarrow} + 2.0 \hat{c}_{3,\downarrow} \hat{c}_{1,\uparrow}^\dagger` is represented by
:code:`sites = [[1,2],[3,1]], sectors=[[0,0], [0,1]], daggers = [[1,0],[0,1]], weights = [1.0, 2.0]`
where we encode :math:`\downarrow,\uparrow` as :code:`0,1`
"""
SpinOperatorArrayDict = dict[int, SpinOperatorArrayTerms]
r"""
A dictionary containing SpinOperatorArrayTerms of different lengths, where the key specifies the lenght,
representing a generic fermionic operator in second quantization.
Version with spin sectors.
"""


def parity(x: Array):
    """
    compute the parity of a permutation (along axis 1)
    vectorized along all leading batch dimensions

    Args:
        x: a numpy array containing the permutation (e.g. obtained from np.argsort)

    Returns:
        False/True depending if an even/odd number of swaps is needed to bring x into order

    this function is equivalent to
    .. code:: python
      import numpy as np
      from functools import partial
      from sympy.combinatorics.permutations import Permutation
      parity = partial(np.apply_along_axis, lambda x: Permutation(x).parity(), -1)

    Example:
        >>> import numpy as np
        >>> from netket.experimental.operator._normal_order_utils import parity
        >>> print(parity(np.array([0,1,2])))
        False
        >>> print(parity(np.array([0,2,1])))
        True
        >>> print(parity(np.array([2,0,1])))
        False
    """
    # see https://github.com/sympy/sympy/blob/96836db06ba6b78103dd4217db53db31502fb2f6/sympy/combinatorics/permutations.py#L115

    n = x.shape[-1]
    A = x[..., :, None] < x[..., None, :]
    mask = np.tril(np.ones((n, n), dtype=bool))
    mask = np.expand_dims(mask, tuple(range(0, A.ndim - 2)))
    return np.bitwise_xor.reduce(A & mask, axis=(-2, -1))


def _prune(sites: Array, daggers: Array, weights: Array) -> OperatorArrayTerms:
    r"""
    remove ĉᵢĉᵢ and ĉᵢ†ĉᵢ† on the same site i

    Args:
        sites: An integer array of size n_terms x n_operators containing the indices i
        daggers: An boolean array of size n_terms x n_operators specifying if the operator is creation/destruction
        weights: An array of size n_terms containing the weight of each term
    Returns:
        (sites, daggers, weights) where the terms with repeated ĉᵢĉᵢ/ĉᵢ†ĉᵢ† have been removed
    """
    mask = ~((np.diff(daggers, axis=-1) == 0) & (np.diff(sites, axis=-1) == 0)).any(
        axis=-1
    )
    return sites[mask], daggers[mask], weights[mask]


def _move(i: Array, j: Array, x: Array, mask: Array = None) -> Array:
    r"""
    move element i after element j on the last axis (batched)
    use the mask parameter to only selectively move some elements in the batch

    Args:
        i: integer array of shape (...,) with values in 0,...,n-1
        j: integer array of shape (...,) with values in 0,...,n-1
        x: array of shape (..., n)
        mask: boolean array of size (...,)

    Returns:
        an array where the i'th element of x has been moved after the jth element if mask is True,
        else x if mask is False

    Example: if x is a 2-dimensional matrix of shape (m,n) and i,j are vectors of size m then,
    in each row k in 0,...,m-1, this removes the element A[k, i[k]] and inserts it after element A[k, j[k]]
    """
    n = x.shape[-1]
    a = np.arange(n)[None]
    masklr = (a < i) | (a > j)
    maskj = a == j
    mask_middle = ~(masklr | maskj)
    x1 = np.roll(x, -1, axis=-1)
    xi = np.take_along_axis(x, i, 1)
    res = masklr * x + mask_middle * x1 + maskj * xi
    if mask is None:
        return res
    return res * mask + (~mask) * x


def _remove(i: Array, j: Array, x: Array) -> Array:
    r"""
    remove element i and j on the last axis (batched)

    Args:
        i: integer array of shape (...,) with values in 0,...,n-1
        j: integer array of shape (...,) with values in 0,...,n-1
        x: array of shape (..., n)
        mask: boolean array of size (...,)

    Returns:
        an array of shape (..., n-2) where the i'th and jth element of x has been removed

    Example: if x is a 2-dimensional matrix of shape (m,n) and i,j are vectors of size m then,
    in each row k in 0,...,m-1, this removes the elements in A[k, i[k]] and A[k,j[k]]
    resulting in a (m,n-2) matrix
    """
    n = x.shape[-1]
    a = np.arange(n)[None]
    maskl = a < i
    maskr = a > j - 2
    mask_middle = ~(maskl | maskr)
    x1 = np.roll(x, -1, axis=-1)
    x2 = np.roll(x, -2, axis=-1)
    return (maskl * x + mask_middle * x1 + maskr * x2)[..., :-2]


def _move_daggers_left(
    sites: Array, daggers: Array, weights: Array
) -> tuple[OperatorArrayTerms]:
    r"""
    apply the fermionic anticommutation rules to recursively bring a set of
    fermionic operators (all of same length) into normal ordering (daggers to the left)


    Args:
        sites: An integer array of size n_terms x n_operators containing the indices i
        daggers: An boolean array of size n_terms x n_operators specifying if the operator is creation/destruction
        weights: An array of size n_terms containing the weight of each term

    Returns:
        a tuple of ((sites, daggers, weights), (sites, daggers, weights), ...)
        of operators with the same length, length-2, ..., length 0
    """

    # TODO
    # re-implement non-recursively using Wick thm

    n = daggers.shape[-1]
    if n == 0:
        return ((sites, daggers, weights),)

    sites_ = sites
    daggers_ = daggers
    weights_ = weights

    new_sites_smaller = []
    new_daggers_smaller = []
    new_weights_smaller = []

    while True:
        a = np.arange(n)[None]
        # find leftmost c
        i = np.argmin(daggers_, axis=1, keepdims=True)
        # find next c^\dagger
        j = np.argmax((i < a) & daggers_, axis=1, keepdims=True)

        # now move the c after the c^\dagger if necessary

        do_move = j > i
        if ~do_move.any():
            break

        sign = 1 - 2 * (((j - i) * do_move) % 2).ravel()

        si = np.take_along_axis(sites_, i, 1)
        sj = np.take_along_axis(sites_, j, 1)

        new_sites = _move(i, j, sites_, mask=do_move)
        new_daggers = _move(i, j, daggers_, mask=do_move)
        new_weights = weights_ * sign

        same = ((si == sj) & do_move).ravel()
        new_sites2 = _remove(i[same], j[same], sites_[same])
        new_daggers2 = _remove(i[same], j[same], daggers_[same])
        new_weights2 = -(weights_ * sign)[same]
        new_sites2, new_daggers2, new_weights2 = _prune(
            new_sites2, new_daggers2, new_weights2
        )

        if len(new_sites2) > 0:
            new_sites_smaller.append(new_sites2)
            new_daggers_smaller.append(new_daggers2)
            new_weights_smaller.append(new_weights2)

        # set variables for next iteration
        sites_, daggers_, weights_ = _prune(new_sites, new_daggers, new_weights)

    if len(new_sites_smaller) > 0:
        new_sites_smaller = np.concatenate(new_sites_smaller, axis=0)
        new_daggers_smaller = np.concatenate(new_daggers_smaller, axis=0)
        new_weights_smaller = np.concatenate(new_weights_smaller, axis=0)
        # recursion; TODO collapse first and only run once for each size instead?
        return (sites_, daggers_, weights_), *_move_daggers_left(
            new_sites_smaller, new_daggers_smaller, new_weights_smaller
        )
    else:
        return ((sites_, daggers_, weights_),)


def move_daggers_left(t: OperatorArrayDict) -> OperatorArrayDict:
    r"""
    Apply the fermionic anticommutation rules to recursively bring a set of
    fermionic operators into normal ordering (daggers to the left)

    Args:
        t: a dictionary containing strings of different lengths, each stored as a tuple
        (sites, daggers, weights)

    Returns:
        a new dictionary with normal ordered strings

    """
    d = {}
    for sdw in [x for v in t.values() for x in _move_daggers_left(*v)]:
        k = sdw[0].shape[-1]
        if sdw[-1].size > 0:  # not empty
            di = d.pop(k, None)
            if di is None:
                d[k] = sdw
            else:
                d[k] = tuple(np.concatenate([a, b], axis=0) for a, b in zip(di, sdw))
    return d


def _to_desc_order(sites: Array, daggers: Array, weights: Array) -> OperatorArrayTerms:
    r"""
    Reorder operators (all of same length) such that the ones with the larger site index are to the left
    Assumes the operators are already in normal order (daggers to the left).


    Args:
        sites: An integer array of size n_terms x n_operators containing the indices i
        daggers: An boolean array of size n_terms x n_operators specifying if the operator is creation/destruction
        weights: An array of size n_terms containing the weight of each term

    Returns:
        (sites, daggers, weights) with sites in descending order
    """
    n = daggers.shape[-1]
    if n == 0:
        return sites, daggers, weights
    # check min and max do not over/underflow
    # TODO promote to signed / bigger dtype if necessary
    xl = sites.min() - 1
    xr = sites.max() + 1
    assert (xl < sites.min()).all()
    assert (xr > sites.max()).all()

    # minus because we order descending
    s0 = -daggers * xr - (1 - daggers) * sites
    s1 = -daggers * sites - (1 - daggers) * xl

    perm0 = np.argsort(s0, axis=-1)
    perm1 = np.argsort(s1, axis=-1)
    a = np.arange(len(sites))[:, None]
    sites_desc = sites[a, perm0] * (1 - daggers) + sites[a, perm1] * daggers
    weights_desc = weights * (1 - 2 * (parity(perm0) ^ parity(perm1)))

    # TODO also merge duplicates
    return _prune(sites_desc, daggers, weights_desc)


def to_desc_order(t: OperatorArrayDict) -> OperatorArrayDict:
    r"""
    Reorder operators such that the ones with the larger site index are to the left
    Assumes the operators are already in normal order (daggers to the left).

    Args:
        t: a dictionary containing strings of different lengths, each stored as a tuple
        (sites, daggers, weights)

    Returns:
        a new dictionary with strings in descending order
    """
    # TODO sum duplicates
    return {k: _to_desc_order(*v) for k, v in t.items()}


def to_normal_order(t: OperatorArrayDict) -> OperatorArrayDict:
    r"""
    Apply the fermionic anticommutation rules to recursively bring a set of
    fermionic operators into normal ordering (daggers to the left), then
    reorder operators such that the ones with the larger site index are to the left

    Args:
        t: a dictionary containing strings of different lengths, each stored as a tuple
        (sites, daggers, weights)

    Returns:
        a new dictionary with strings in descending normal order
    """
    return to_desc_order(move_daggers_left(t))


def _split_spin_sectors_helper(
    sites: Array,
    daggers: Array,
    weights: Array,
    n_orbitals: int,
    n_spin_subsectors: int,
) -> SpinOperatorArrayTerms:
    n_ops = sites.shape[1]
    if n_ops == 0:
        return sites, np.zeros_like(sites), daggers, weights
    L = np.arange(n_spin_subsectors) * n_orbitals
    R = np.arange(1, n_spin_subsectors + 1) * n_orbitals
    # n_terms x n_ops x n_spin_subsectors
    sectors_mask = (sites[..., None] >= L) & (sites[..., None] < R)
    sectors = np.einsum("...i,i", sectors_mask, np.arange(n_spin_subsectors)).astype(
        np.int32
    )
    sites = sites - sectors * n_orbitals
    return sites, sectors, daggers, weights


def split_spin_sectors(
    d: OperatorArrayDict, n_orbitals: int, n_spin_subsectors: int
) -> SpinOperatorArrayDict:
    r"""
    Split global site indices into spin sector and index within sector

    Args:
        d: { size : (sites, daggers, weights) }
        n_orbitals: number of orbitals (assumed to be the same for each sector)
        n_spin_subsectors: number of spin sectors
    Returns:
        A dictionary { size : (sites, sectors, daggers, weights) }
        where the site indices have been transformed into the spin sectors and index within the sector
    """
    return {
        k: _split_spin_sectors_helper(*v, n_orbitals, n_spin_subsectors)
        for k, v in d.items()
    }


def _merge_spin_sectors_helper(
    sites: Array, sectors: Array, daggers: Array, weights: Array, n_orbitals: int
) -> OperatorArrayTerms:
    return sites + sectors * n_orbitals, daggers, weights


def merge_spin_sectors(d: SpinOperatorArrayDict, n_orbitals) -> OperatorArrayDict:
    r"""
    Merge spin sector and index within sector into global site index

    Args:
        d: { size : (sites, sectors, daggers, weights) }
        n_orbitals: number of orbitals (assumed to be the same for each sector)

    Returns:
        A dictionary { size : (sites, daggers, weights) }
        where the indices and sectors have been transformed into global site indices
    """
    return {k: _merge_spin_sectors_helper(*v, n_orbitals) for k, v in d.items()}


def to_normal_order_sector(
    t: SpinOperatorArrayDict, n_spin_subsectors: int, n_orbitals: int
) -> SpinOperatorArrayDict:
    r"""convert to normal order with higher sector to the left

    Args:
        t: a dictionary containing strings of different lengths, each stored as a tuple
        (sites, sectors, daggers, weights)
        n_spin_subsectors: number of spin sectors
        n_orbitals: number of orbitals (assumed to be the same for each sector)

    Returns:
        a new dictionary with strings in descending normal order

    """
    return split_spin_sectors(
        to_normal_order(merge_spin_sectors(t, n_orbitals)),
        n_orbitals,
        n_spin_subsectors,
    )
