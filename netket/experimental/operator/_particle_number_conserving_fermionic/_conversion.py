# utilities to convert from / to FermionOperator2nd/FermionOperator2ndJax


import numpy as np

import jax.numpy as jnp

from netket.jax import COOArray
from netket.utils.types import Array

from .._normal_order_utils import (
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
