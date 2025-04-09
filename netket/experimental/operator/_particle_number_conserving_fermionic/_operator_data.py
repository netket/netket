# utilities to prepare the internal datastructures

from functools import partial
import numpy as np

import jax
import jax.numpy as jnp

from netket.jax import COOTensor
from netket.utils.optional_deps import import_optional_dependency


def len_helper(s):
    if isinstance(s, tuple):
        return len(s)
    else:
        return 1


def _prepare_data_helper(sites_destr, sites_create, weights, n_orbitals, sparse_=True):

    sparse = import_optional_dependency("sparse")

    # we encode sites_create==sites_destr by passing sites_create=None
    is_diagonal = sites_create is None

    n_terms, half_n_ops = sites_destr.shape
    assert weights.shape == (n_terms,)
    if not is_diagonal:
        assert sites_destr.shape == sites_create.shape

    if half_n_ops == 0:  # constant
        assert n_terms == 1
        index_array = jnp.zeros((), dtype=np.int32)
        create_array = jnp.zeros((1, 1, 0), dtype=np.int32)
        weight_array = jnp.array(weights, dtype=weights.dtype)
    elif is_diagonal:
        assert sites_destr.max() < n_orbitals
        index_array = None
        create_array = None
        # use sparse.COO to sort since COOTensor expects sorted indices
        # TODO do it inside COOTensor
        tmp = sparse.COO(sites_destr.T, weights, (n_orbitals,) * (half_n_ops))
        weight_array = COOTensor(
            jnp.asarray(tmp.coords.T),
            jnp.asarray(tmp.data),
            (n_orbitals,) * (half_n_ops),
        )
        if not sparse_:
            weight_array = weight_array.todense()
    else:
        assert sites_destr.max() < n_orbitals
        assert sites_create.max() < n_orbitals

        ### simple, inefficient version
        # destr_unique = np.unique(sites_destr, axis=0)
        # nper = np.zeros(len(destr_unique), dtype=int)
        # for i, d in enumerate(destr_unique):
        #     nper[i] = (d[None] == sites_destr).all(axis=-1).sum()
        ###
        A = sparse.COO(np.concatenate([sites_destr, sites_create], axis=1).T, weights)
        axes_create = tuple(range(A.ndim // 2, A.ndim))
        n_destr = (A != 0).sum(axes_create)
        destr_unique = n_destr.coords.T
        nper = n_destr.data
        ###

        # we pad with zeros, so we take create_array and weight_array of size nunique+1
        # (where the 0th element is the padding)
        # and put zeros in the index_array, for terms which dont exist

        ### simple, inefficient version
        # nmax = int(nper.max())
        # nunique = len(destr_unique)
        # create_array = jnp.zeros((1 + nunique, nmax, half_n_ops), dtype=np.int32)
        # weight_array = jnp.zeros((1 + nunique, nmax), dtype=weights.dtype)
        # for i, d in enumerate(destr_unique):
        #     mask = (sites_destr == d).all(axis=-1)
        #     weight_array = weight_array.at[i + 1, : nper[i]].set(weights[mask])
        #     create_array = create_array.at[i + 1, : nper[i]].set(sites_create[mask])
        ###
        # select only nonzero destr rows
        B = A[tuple(destr_unique.T)]
        # create an arange for every row
        row_ind, *create_indices = B.coords
        row_start = np.where(np.diff(row_ind, prepend=-1))
        a = np.arange(B.nnz)
        compressed_col_ind = a - np.repeat(a[row_start], nper)
        # +1 because of the padding
        new_coords = np.vstack([row_ind + 1, compressed_col_ind])
        weight_array = jnp.asarray(sparse.COO(new_coords, B.data).todense())
        create_array = jnp.concatenate(
            [sparse.COO(new_coords, c).todense()[..., None] for c in create_indices],
            axis=-1,
        )
        ###
        # destr_unique should be already sorted at this point (in np.unique / sparse.COO)
        index_array = COOTensor(
            jnp.asarray(destr_unique),
            jnp.arange(1, len(destr_unique) + 1),
            (n_orbitals,) * (half_n_ops),
        )
        if not sparse_:
            index_array = index_array.todense()

    return index_array, create_array, weight_array


def prepare_data_diagonal(sites_destr, weights, n_orbitals, **kwargs):
    r"""

    Prepare the custom sparse internal data for ParticleNumberConservingFermioperator2ndJax, for the diagonal part of the operator
    of strings of a fixed length

    Assume we are given a sequence of equal-length normal ordered strings \sum_i w_i c_{a_i1}^\dagger ... c_{a_iN}^\dagger c_{a_i1} ... c_{a_iN}
    with 2N fermionic operators, with larger indices to the left: a_i1 >=...>=a_iN

    Please refer to the docstring of prepare_data for a more complete explanation of the storage format.

    We can treat the diagonal opeators separately to the non-diagonal ones in a more efficient way:
        index_array: None
        create_array: None
        weight_array: shape (n,)*N  is indexed directly with the list of destruction operators (given by b above)

    Args:
        sites_destr: a matrix containing the indices of c^dagger/c for every string [[a_i1, ..., a_iN]] ()
        weights: array of the corresponding weights [w_i]
        n_orbitals: number of orbitals n
        sparse_: whether to store weight_array in dense or sparse
    Returns:
        A tuple (index_array, create_array, weight_array) as defined above

    """
    return _prepare_data_helper(sites_destr, None, weights, n_orbitals, **kwargs)


def prepare_data(sites, weights, n_orbitals, **kwargs):
    r"""
    Prepare the custom sparse internal data for ParticleNumberConservingFermioperator2ndJax
    of strings of a fixed length

    It is given by a 3-tuple for every length N of string in normal ordering (containg 2N fermionic operators),

    Assume we are given a sequence of equal-length normal ordered strings \sum_i w_i c_{a_i1}^\dagger ... c_{a_iN}^\dagger c_{b_i1} ... c_{b_iN}
    with 2N fermionic operators, with larger indices to the left: a_i1 >=...>=a_iN, b_i1 >=...>=b_iN

    Here a \in {1..n} contains the indices of the c^\dagger, b the indices of the c operators and w the corresponding weight of every string.
    where n is the number of orbitals.

    The 3-tuple for these operators is given by
        index_array: shape (n,)*N+(n_max,) contains list of integer indices for all the strings for a given list of destruction operators (given by b above)
                     can be stored either in a dense, or sparse format; is padded to the maximum number of operators n_max for a given sequence of destruction ops (b)
                     Since the operators are assumed to be in normal order (larger sites to the left) only the lower triangular part is used.
        create_array: shape (n_ops+1,)+(N,) for every index from index_array contains creation operators of the corresponding term (given by a above)
        weight_array: shape (n_ops+1,)+(1,) for every index from index_array contains the weight of the corresponding term
                      The 0 weight for the padding is stored as the last element.

    Then for a given basis state |b_1,...,b_m> (where b_j indicates the occupied orbitals, for a fixed number of electrons m) we find all the connected elements
    by taking all m choose N combinations of occupied orbitals to be destroyed as index for index_array, excluding the strings which try to destroy an empty orbital.


    Args:
        sites: a matrix containing the indices of c^dagger and c for every string [[a_i1, ..., a_iN, b_i1, ..., b_iN]]
        weights: array of the corresponding weights [w_i]
        n_orbitals: number of orbitals n
        sparse_: wether to store index_array in dense or sparse
    Returns:
        A tuple (index_array, create_array, weight_array) as defined above

    """

    # sites is an array (n_terms, n_ops) containing the sites
    # of terms in normal order (daggers to left, desc order)
    #
    # weights is of shape (n_terms,)
    n_terms, n_ops = sites.shape
    assert n_ops % 2 == 0
    sites_destr = sites[:, : n_ops // 2]
    sites_create = sites[:, n_ops // 2 :]
    return _prepare_data_helper(
        sites_destr, sites_create, weights, n_orbitals, **kwargs
    )


def split_diag_offdiag(sites, weights):
    """
    Split the diagonal and off-diagonal terms
    """
    n_terms, n_ops = sites.shape
    assert weights.shape == (n_terms,)
    assert n_ops % 2 == 0

    idestr = sites[:, : (n_ops // 2)]
    icreate = sites[:, (n_ops // 2) :]

    is_diag = (idestr == icreate).all(axis=-1)
    diag_sites = idestr[is_diag]
    diag_weights = weights[is_diag]
    offdiag_sites = sites[~is_diag]
    offdiag_weights = weights[~is_diag]
    return (diag_sites, diag_weights), (offdiag_sites, offdiag_weights)


def prepare_operator_data_from_coords_data_dict(coords_data_dict, n_orbitals, **kwargs):
    r"""
    Prepare the custom sparse internal data for ParticleNumberConservingFermioperator2ndJax

    of a string of operators \sum_N \sum_i w_i^{(N)} c_{a_i1^{(N)}}^\dagger ... c_{a_iN^{(N)}}^\dagger c_{b_i1^{(N)}} ... c_{b_iN^{(N)}}
    in descending order a_i1 >=...>=a_iN, b_i1 >=...>=b_iN.

    Please refer to the docstring of prepare_data, prepare_data_diagonal for a more complete explanation of the storage format.

    Args:
        coords_data_dict: A dictionary {N: (sites, weights)}
            where for every length N
                sites is a matrix containing the stacked indices [[a_i1^{(N)}, ... a_iN^{(N)}, b_i1^{(N)}, ..., b_iN^{(N)}]]
                of the c^\dagger and c
                weights is a vector containing the corresponding weights [w_i^{(N)}]
        n_orbitals: number of orbitals
    Returns:
        A dictionary {'diag': { N:  (None, None, weight_array)}, 'offdiag': { N : (index_array, create_array, weight_array)}}
        containing the sparse representation for every lenght N of strings c_{a_i1}^\dagger ... c_{a_iN}^\dagger c_{b_i1} ... c_{b_iN}
    """
    data_offdiag = {}
    data_diag = {}
    for k, v in coords_data_dict.items():
        sw_diag, sw_offdiag = split_diag_offdiag(*v)
        if len(sw_diag[-1]) > 0:
            data_diag[k] = prepare_data_diagonal(*sw_diag, n_orbitals, **kwargs)
        if len(sw_offdiag[-1]) > 0:
            data_offdiag[k] = prepare_data(*sw_offdiag, n_orbitals, **kwargs)
    data = {"diag": data_diag, "offdiag": data_offdiag}
    return data


def sparse_arrays_to_coords_data_dict(ops):
    """
    Split dictionary values of sparse arrays into indices and data

    (optionally) supports spin sectors in the key
    """

    def _unpack(k, v):
        if isinstance(k, tuple):
            k, _ = k  # _ are the spin sectors
        if k == 0:
            return np.zeros((1, 0), dtype=int), np.array([v])
        else:
            return v.coords.T, v.data

    return {k: _unpack(k, v) for k, v in ops.items()}


def collect_ops(operators):
    r"""
    sum operators with the same number of c/c^\dagger

    Args:
        operators: a list of scalars / sparse matrices / dense matrices
    Returns:
        A dictionary of sparse matrices, one for each lenght of  c/c^\dagger
    """

    sparse = import_optional_dependency("sparse")

    ops = {}
    for A in operators:
        if isinstance(A, sparse.COO):
            k = A.ndim
            if A.shape == ():
                A = A.fill_value
            else:
                assert A.fill_value == 0
        elif jnp.isscalar(A) or (hasattr(A, "__array__") and A.ndim == 0):
            A = np.asarray(A)
            k = 0
        elif hasattr(A, "__array__"):
            A = sparse.COO.from_numpy(np.asarray(A))
            k = A.ndim
        else:
            raise NotImplementedError
        Ak = ops.pop(k, None)
        if Ak is not None:
            ops[k] = Ak + A
        else:
            ops[k] = A
    return ops


def prepare_operator_data_from_coords_data_dict_spin(coords_data_sectors, n_orbitals):
    # version with sectors
    _cond = lambda s: s == () or len_helper(s[0]) == 1
    coords_data_same = {
        (k, s): v for (k, s), v in coords_data_sectors.items() if _cond(s)
    }
    coords_data_mixed = {
        (k, s): v for (k, s), v in coords_data_sectors.items() if not _cond(s)
    }

    operator_data = prepare_operator_data_from_coords_data_dict(
        coords_data_same, n_orbitals
    )
    # process mixed terms
    data_diag_mixed = {}
    data_offdiag_mixed = {}
    for k, v in coords_data_mixed.items():
        sw_diag, sw_offdiag = split_diag_offdiag(*v)
        if len(sw_diag[-1]) > 0:
            data_diag_mixed[k] = prepare_data_diagonal(
                *sw_diag, n_orbitals, sparse_=False
            )
        if len(sw_offdiag[-1]) > 0:
            data_offdiag_mixed[k] = prepare_data(*sw_offdiag, n_orbitals, sparse_=False)
    operator_data = {
        **operator_data,
        "mixed_diag": data_diag_mixed,
        "mixed_offdiag": data_offdiag_mixed,
    }
    return operator_data


def sites_daggers_weights_to_sparse(sites, daggers, weights, n_orbitals):

    sparse = import_optional_dependency("sparse")

    n = daggers.shape[-1]
    assert n % 2 == 0
    assert (daggers[:, : n // 2] == 1).all()
    assert (daggers[:, n // 2 :] == 0).all()
    # TODO cutoff?
    return sparse.COO(sites.T, weights, shape=(n_orbitals,) * n)


def _insert_append_helper(d, k, s, o, cutoff):
    # check if an element with the same matrix but different sectors exist
    # if yes append to the list of sectors
    # else insert new element into the dict

    sparse = import_optional_dependency("sparse")

    for (k2, s2), o2 in d.items():
        same_number_of_sectors = (s == () and s2 == ()) or (
            len(s2) > 0 and len(s) > 0 and len_helper(s2[0]) == len_helper(s[0])
        )
        if (
            same_number_of_sectors
            and k == k2  # same_number_of_fermionic_operators
            and sparse.abs(o - o2).max() < cutoff  # same_matrix
        ):
            d[k, s2 + s] = d.pop((k2, s2))
            break
    else:
        d[k, s] = o


def to_coords_data_sector(tno_sector, n_spin_subsectors, n_orbitals, cutoff=1e-11):
    r"""
    Args:
        tno_sector: a list of tuples [(sites, sectors, daggers, weights)]
                    of terms in normal order with higher sectors on the left

    Returns: a dict {(k, sectors) : (indices, data)}
             where k is the number of c/c^\dagger,
             sectors are the spin sectors acted on,
             indices contains the sites inside of each sector
             and data contains the weights
    """

    operators_sector = {}

    for k, (sites, sectors, daggers, weights) in tno_sector.items():
        for i in range(n_spin_subsectors):
            if not (((2 * daggers - 1) * (sectors == i)).sum(axis=-1) == 0).all():
                raise ValueError  # does not conserve particle number per sector

        sector_count = jax.vmap(partial(jnp.bincount, length=n_spin_subsectors))(
            sectors
        )

        # merge sectors which have same sparse matrix

        if k == 0:
            operators_sector[0, ()] = weights.reshape(())
        elif k == 2:
            # at this point we know there is only one sector this acts on
            sector = sectors[:, 0]  # = sectors[:, 1]
            for i in np.unique(sector):
                m = sector == i
                o = sites_daggers_weights_to_sparse(
                    sites[m], daggers[m], weights[m], n_orbitals=n_orbitals
                )
                _insert_append_helper(operators_sector, k, (i,), o, cutoff)
        elif k == 4:
            # at this point we know that n_sectors_acting_on \in 1,2
            n_sectors_acting_on = np.count_nonzero(sector_count, axis=-1)

            # all same sector
            m_same = n_sectors_acting_on == 1
            sector = sectors[:, 0]
            for i in np.unique(sector[m_same]):
                m = (sector == i) & m_same
                o = sites_daggers_weights_to_sparse(
                    sites[m], daggers[m], weights[m], n_orbitals=n_orbitals
                )
                _insert_append_helper(operators_sector, k, (i,), o, cutoff)

            m_different = ~m_same
            sector = sectors[:, :2]
            # i > j because we made it normal order (with site shifted by N*spin) above
            for ij in np.unique(sector[m_different], axis=0):
                m = (sector == ij[None]).all(axis=-1) & m_different
                # minus sign because in the operator (_get_conn_padded_interaction_up_down) we assume it's swaped to (assuming σ>ρ)
                # cσ^† cσ cρ^† cρ = - cσ^† cρ^† cσ cρ
                o = -sites_daggers_weights_to_sparse(
                    sites[m], daggers[m], weights[m], n_orbitals=n_orbitals
                )
                _insert_append_helper(operators_sector, k, (tuple(ij),), o, cutoff)
        else:
            raise NotImplementedError

    return sparse_arrays_to_coords_data_dict(operators_sector)
