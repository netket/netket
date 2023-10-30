import sparse
import numpy as np

import jax
import jax.numpy as jnp

from netket.utils.optional_deps import import_optional_dependency
from netket.experimental.hilbert import SpinOrbitalFermions
from ._fermion_operator_2nd import FermionOperator2nd


def compute_pyscf_integrals(mol, mo_coeff):
    t_ij_ao = jnp.asarray(mol.intor("int1e_kin") + mol.intor("int1e_nuc"))
    v_ijkl_ao = jnp.asarray(mol.intor("int2e").transpose(0, 2, 3, 1))
    c = jnp.asarray(mo_coeff.T)
    t_ij_mo = jax.jit(jnp.einsum, static_argnums=0)("ij,ai,bj->ab", t_ij_ao, c, c)
    v_ijkl_mo = jax.jit(jnp.einsum, static_argnums=0)(
        "ijkl,ai,bj,ck,dl->abcd", v_ijkl_ao, c, c, c, c
    )
    const = jnp.asarray(float(mol.energy_nuc()))
    return const, t_ij_mo, v_ijkl_mo


def spinorb_from_spatial_sparse_coo2(tij_sparse, interleave=False, spin_values=[0, 1]):
    # Σ_ijσ t_ij c†_iσ c_jσ
    # for σ ∈ spin_values
    # interleave=True -> 2i+spin
    # interleave=False -> n*spin+i

    if isinstance(spin_values, int):
        spin_values = list(spin_values)
    # assert spin_values in [[0], [1], [0,1]]

    assert isinstance(tij_sparse, sparse.COO)
    n = tij_sparse.shape[0]

    if interleave:
        t = lambda x: 2 * x
        tp1 = lambda x: 2 * x + 1
    else:
        t = lambda x: x
        tp1 = lambda x: x + n

    i, j = tij_sparse.coords
    a = tij_sparse.data
    a2 = []
    i2 = []
    j2 = []
    if 0 in spin_values:
        a2.append(a)
        i2.append(t(i))
        j2.append(t(j))
    if 1 in spin_values:
        a2.append(a)
        i2.append(tp1(i))
        j2.append(tp1(j))
    i2 = np.concatenate(i2)
    j2 = np.concatenate(j2)
    a2 = np.concatenate(a2)
    return sparse.COO(np.array([i2, j2]), a2, shape=(2 * n, 2 * n))


def spinorb_from_spatial_sparse_coo4(
    vijkl_sparse, interleave=False, _order_preserving=False
):
    # Σ_ijklμσ v_ijkl c†_iμ c†_jσ c_kμ c_lσ
    # interleave=True -> 2i+spin
    # interleave=False -> n*spin+i
    assert isinstance(vijkl_sparse, sparse.COO)
    n = vijkl_sparse.shape[0]

    if interleave:
        t = lambda x: 2 * x
        tp1 = lambda x: 2 * x + 1
    else:
        t = lambda x: x
        tp1 = lambda x: x + n

    p, q, r, s = vijkl_sparse.coords
    b = vijkl_sparse.data

    if _order_preserving:
        # swap i/j and k/l
        a2 = np.concatenate([-b, -b, b, b])
        p2 = np.concatenate([tp1(q), tp1(p), t(p), tp1(p)])
        q2 = np.concatenate([t(p), t(q), t(q), tp1(q)])
        r2 = np.concatenate([tp1(r), tp1(s), t(r), tp1(r)])
        s2 = np.concatenate([t(s), t(r), t(s), tp1(s)])
    else:
        # same as openfermion
        a2 = np.concatenate([b, b, b, b])
        p2 = np.concatenate([t(p), tp1(p), t(p), tp1(p)])
        q2 = np.concatenate([tp1(q), t(q), t(q), tp1(q)])
        r2 = np.concatenate([tp1(r), t(r), t(r), tp1(r)])
        s2 = np.concatenate([t(s), tp1(s), t(s), tp1(s)])

    return sparse.COO(
        np.array([p2, q2, r2, s2]), a2, shape=(2 * n, 2 * n, 2 * n, 2 * n)
    )


def to_desc_order_sparse(vijkl_sparse, cutoff, set_zero_same=True):
    # !! use this only after adding spin, spinorb_from_spatial_sparse will be wrong
    # because the (implicitly assumed) symmetries of the tensor are not conserved

    # assume (1,1,0,0) are already index order (daggers are left/ij)
    # swap i/j k/l so that i>j and k>l

    # now swap the larger one to the left, will cause lots of them to cancel
    i, j, k, l = vijkl_sparse.coords
    a = vijkl_sparse.data.copy()
    a[np.where(i < j)] *= -1
    a[np.where(k < l)] *= -1
    if set_zero_same:
        # set to zero all those where we try to create / destroy two on the same orbital
        a[np.where(i == j)] *= 0
        a[np.where(k == l)] *= 0
    new_coords = np.array(
        [np.maximum(i, j), np.minimum(i, j), np.maximum(k, l), np.minimum(k, l)]
    )
    # use coo to merge same indices
    vijkl_sparse2 = sparse.COO(new_coords, a, shape=vijkl_sparse.shape)
    # we might have some new almost zeros from the cancellations, make sure they are 0
    new_coords2 = vijkl_sparse2.coords
    a2 = vijkl_sparse2.data
    mask = np.abs(a2) > cutoff
    return sparse.COO(new_coords2[:, mask], a2[mask], shape=vijkl_sparse.shape)


def spinorb_from_spatial_sparse(tij_sparse, vijkl_sparse, interleave=False):
    return spinorb_from_spatial_sparse_coo2(
        tij_sparse, interleave
    ), spinorb_from_spatial_sparse_coo4(vijkl_sparse, interleave)


def pyscf_molecule_to_arrays(mol, mo_coeff=None):
    if mo_coeff is None:
        pyscf = import_optional_dependency("pyscf", descr="pyscf_molecule_to_arrays")

        mf = pyscf.scf.HF(mol).run()
        mo_coeff = mf.mo_coeff
    return compute_pyscf_integrals(mol, mo_coeff)


def arrays_to_terms(
    const, tij_sparse, vijkl_sparse, term_conj2=(1, 0), term_conj4=(1, 0, 1, 0)
):
    ij = np.array(
        [
            tij_sparse.coords[0],
            np.ones_like(tij_sparse.coords[0], dtype=int) * term_conj2[0],
            tij_sparse.coords[1],
            np.ones_like(tij_sparse.coords[1], dtype=int) * term_conj2[1],
        ]
    ).T.reshape(-1, 2, 2)

    ijkl = np.array(
        [
            vijkl_sparse.coords[0],
            np.ones_like(vijkl_sparse.coords[0], dtype=int) * term_conj4[0],
            vijkl_sparse.coords[1],
            np.ones_like(vijkl_sparse.coords[1], dtype=int) * term_conj4[1],
            vijkl_sparse.coords[2],
            np.ones_like(vijkl_sparse.coords[2], dtype=int) * term_conj4[2],
            vijkl_sparse.coords[3],
            np.ones_like(vijkl_sparse.coords[3], dtype=int) * term_conj4[3],
        ]
    ).T.reshape(-1, 4, 2)
    terms = ij.tolist() + ijkl.tolist()
    weights = tij_sparse.data.tolist() + vijkl_sparse.data.tolist()
    return terms, weights, const


def operator_from_arrays(
    const,
    tij_sparse,
    vijkl_sparse,
    n_electrons,
    hi=None,
    cls=FermionOperator2nd,
    term_conj2=(1, 0),
    term_conj4=(1, 0, 1, 0),
):
    """
    H = const + Σ tᵢⱼ cᵢ†cⱼ + Σ vᵢⱼₖₗ cᵢ†cⱼcₖ†cₗ
    use term_conj4=(1,1,0,0) if you want cᵢ†cⱼ†cₖcₗ instead.
    use a zero matrix/tensor of the correct shape for tij and/or vijkl if you only want to use one of them
    """
    if hi is None:
        if isinstance(n_electrons, tuple):
            assert len(n_electrons) == 2
            hi = SpinOrbitalFermions(
                n_orbitals=tij_sparse.shape[0] // 2, s=1 / 2, n_fermions=n_electrons
            )
        else:
            hi = SpinOrbitalFermions(
                n_orbitals=tij_sparse.shape[0], n_fermions=n_electrons
            )
    terms, weights, constant = arrays_to_terms(
        const, tij_sparse, vijkl_sparse, term_conj2=term_conj2, term_conj4=term_conj4
    )
    return cls(hi, terms, weights, constant)


def FermiOperator2nd_from_pyscf_molecule(
    mol, mo_coeff=None, cutoff=1e-11, cls=FermionOperator2nd
):
    """defaults to molecular orbitals
    !! factor 0.5 is added here
    """
    E_nuc, tij, vijkl = pyscf_molecule_to_arrays(mol, mo_coeff)
    tij_sparse = sparse.COO.from_numpy(tij * (np.abs(tij) >= cutoff))
    vijkl_sparse = sparse.COO.from_numpy(vijkl * (np.abs(vijkl) >= cutoff))
    tij_sparse_spin, vijkl_sparse_spin = spinorb_from_spatial_sparse(
        tij_sparse, vijkl_sparse
    )
    vijkl_sparse_spin = to_desc_order_sparse(vijkl_sparse_spin, cutoff)
    ha = operator_from_arrays(
        E_nuc,
        tij_sparse_spin,
        0.5 * vijkl_sparse_spin,
        mol.nelec,
        term_conj4=(1, 1, 0, 0),
        cls=cls,
    )
    # TODO run setup and set _max_conn_size here estimating it analytially
    return ha
