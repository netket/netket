# utilities to convert from / to FermionOperator2nd/FermionOperator2ndJax


import numpy as np

import jax.numpy as jnp

from netket.jax import COOArray
from netket.utils.types import Array

from netket._src.operator.normal_order_utils import (
    split_spin_sectors,
    OperatorArrayDict,
    SpinOperatorArrayDict,
)

from netket.operator._fermion2nd.utils import OperatorTermsList, OperatorWeightsList


# TODO merge this with fermionoperator2nd prepare_terms_list
def fermiop_to_pnc_format_helper(
    terms: OperatorTermsList, weights: OperatorWeightsList
) -> OperatorArrayDict:
    r"""
    helper function to convert OperatorTermsList and OperatorWeightsList to OperatorArrayDict

    Args:
        terms: terms as specified in FermionOperator2nd/FermionOperator2ndJax
        weights: a list of weights
    Returns:
        a dictionary {k: (sites, daggers, weights)}
        where for every set of operators of length k
            sites: (n_terms, k) matrix containing the indices of the c/c^\dagger operators
            daggers: (n_terms,k), matrix storing c/c^\dagger as 0/1
            weights: (n_terms,) vector of corresponding weights
    """
    out = {}
    for t, w in zip(terms, weights):
        if len(t) == 0:  # constant
            out[0] = (
                np.zeros((1, 0), dtype=np.int32),
                np.zeros((1, 0), dtype=np.int8),
                np.array([w]),
            )
        else:
            sites, daggers = np.array(t).T
            l = len(daggers)
            assert l % 2 == 0
            assert 2 * daggers.sum() == l
            tl, dl, wl = out.get(l, ([], [], []))
            out[l] = tl + [sites,], dl + [daggers,], wl + [w,]  # fmt: skip
    return {
        k: (
            jnp.array(v[0], dtype=np.int32),
            jnp.array(v[1], dtype=np.int8),
            jnp.array(v[2]),
        )
        for k, v in out.items()
    }


def fermiop_to_pnc_format_spin_helper(
    terms: OperatorTermsList,
    weights: OperatorWeightsList,
    n_orbitals: int,
    n_spin_subsectors: int,
) -> SpinOperatorArrayDict:
    r"""
    convert OperatorTermsList and OperatorWeightsList to SpinOperatorArrayDict
    """
    # output: { size : (sites, sectors, daggers, weights) }
    return split_spin_sectors(
        fermiop_to_pnc_format_helper(terms, weights),
        n_orbitals,
        n_spin_subsectors,
    )


def pnc_format_to_fermiop_helper(
    index_array: Array, create_array: Array, weight_array: Array
) -> tuple[OperatorTermsList, OperatorWeightsList]:
    r"""
    convert OperatorArrayTerms to OperatorTermsList and OperatorWeightsList
    """
    if index_array is None:  # diagonal
        if weight_array.ndim == 0:  # const
            return np.array([()], dtype=np.int32), np.array(weight_array)
        else:
            if not isinstance(weight_array, COOArray):
                weight_array = COOArray.fromdense(weight_array)
            destr = np.array(weight_array.coords.T)
            weights = np.array(weight_array.data)
            sites = np.concatenate([destr, destr], axis=-1)
    else:
        if index_array.ndim == 0:  # const
            return np.array([()], dtype=np.int32), np.array(weight_array)
        if not isinstance(index_array, COOArray):
            index_array = COOArray.fromdense(index_array)
        ind = np.array(index_array.data)
        destr = np.array(index_array.coords.T[:, None, :])
        create = create_array[ind]
        destr = np.broadcast_to(destr, create.shape)
        weights = weight_array[ind]
        sites = np.concatenate([destr, create], axis=-1)

    # flatten
    weights = weights.reshape(-1)
    sites = sites.reshape(-1, sites.shape[-1])

    daggers = np.zeros_like(sites)
    daggers[:, : daggers.shape[1] // 2] = 1
    terms = np.concatenate([sites[..., None], daggers[..., None]], axis=-1)

    return terms, weights


def pnasc_format_to_fermiop_helper(
    index_array: Array,
    create_array: Array,
    weight_array: Array,
    sectors: tuple,
    n_orbitals: int,
) -> tuple[OperatorTermsList, OperatorWeightsList]:
    r"""
    Convert spin-conserving PNC format to FermionOperator format

    Args:
        index_array: destruction operator indices (orbital indices within sectors)
        create_array: creation operator indices (orbital indices within sectors)
        weight_array: weights
        sectors: which spin sectors are acted upon, e.g.:
            - () for constant term
            - (0,) for single sector 0
            - (0, 1) for multiple sectors with same matrix
            - ((0, 1),) for mixed sectors (create in 0, destroy in 1)
        n_orbitals: number of orbitals per sector

    Returns:
        (terms, weights) in FermionOperator2nd format with global site indices
    """
    # First convert to terms using the existing helper
    terms_base, weights_base = pnc_format_to_fermiop_helper(
        index_array, create_array, weight_array
    )

    if len(terms_base) == 0 or (len(terms_base) > 0 and terms_base.shape[1] == 0):
        # Empty or constant term
        return terms_base, weights_base

    # Now handle sector encoding: site_global = site_orbital + sector * n_orbitals
    all_terms = []
    all_weights = []

    if sectors == ():
        # Constant term, no sector adjustment needed
        return terms_base, weights_base

    # Check if this is a same-sector or mixed-sector case
    if len(sectors) > 0 and isinstance(sectors[0], (int, np.integer)):
        # Same sector: all operators in the same sector
        # sectors could be (0,) or (0, 1, 2) if multiple sectors share the same matrix
        for sector in sectors:
            terms_copy = terms_base.copy()
            terms_copy[:, :, 0] += sector * n_orbitals
            all_terms.append(terms_copy)
            all_weights.append(weights_base)
    else:
        # Mixed sector: ((i, j),) or ((i, j), (k, l))
        # The encoding transforms the original term to match pattern: c_i,σ^ c_j,ρ^ c_k,ρ c_l,σ
        # where sectors contains [σ, ρ]
        # During encoding, destruction operators are swapped and a minus sign is applied
        # We need to reverse this transformation during decoding
        for sector_pair in sectors:
            terms_copy = terms_base.copy()
            weights_copy = weights_base.copy()
            n_ops = terms_copy.shape[1]

            if n_ops == 4:
                # The stored format has:
                # - Creation orbitals [i, j] in sectors [σ, ρ]
                # - Destruction orbitals [k, l] in sectors [ρ, σ] (swapped during encoding!)
                # - Weight negated during encoding

                # To decode to the original term, we need to:
                # 1. Swap destruction operators back: [k, l] in [ρ, σ] → [l, k] in [σ, ρ]
                # 2. Negate the weight

                sector_create1, sector_create2 = sector_pair  # σ, ρ

                # Apply sector offsets
                # Creation operators stay as-is
                terms_copy[:, 0, 0] += sector_create1 * n_orbitals  # first creation in sector σ
                terms_copy[:, 1, 0] += sector_create2 * n_orbitals  # second creation in sector ρ

                # Destruction operators need to be swapped back
                # Stored as [k, l] in sectors [ρ, σ]
                # Need to convert to [l, k] in sectors [σ, ρ]
                # So: operator at position 2 goes to sector σ (should be at position 3 originally)
                #     operator at position 3 goes to sector ρ (should be at position 2 originally)
                # Therefore we swap the sector assignments:
                terms_copy[:, 2, 0] += sector_create1 * n_orbitals  # first destruction → sector σ
                terms_copy[:, 3, 0] += sector_create2 * n_orbitals  # second destruction → sector ρ

                # Negate weight to reverse the encoding transformation
                weights_copy = -weights_copy
            else:
                # General case (shouldn't happen based on current implementation)
                sector_create, sector_destroy = sector_pair
                half_n_ops = n_ops // 2
                terms_copy[:, :half_n_ops, 0] += sector_create * n_orbitals
                terms_copy[:, half_n_ops:, 0] += sector_destroy * n_orbitals

            all_terms.append(terms_copy)
            all_weights.append(weights_copy)

    if len(all_terms) > 1:
        terms = np.concatenate(all_terms, axis=0)
        weights = np.concatenate(all_weights, axis=0)
    else:
        terms = all_terms[0]
        weights = all_weights[0]

    return terms, weights
