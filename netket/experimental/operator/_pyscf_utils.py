import numpy as np

import jax
import jax.numpy as jnp

from netket.utils.optional_deps import import_optional_dependency

from ._normal_order_utils import parity


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
    sparse = import_optional_dependency(
        "sparse", descr="spinorb_from_spatial_sparse_coo2"
    )

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
    sparse = import_optional_dependency(
        "sparse", descr="spinorb_from_spatial_sparse_coo4"
    )

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
    sparse = import_optional_dependency("sparse", descr="to_desc_order_sparse")

    # assume (1,1,0,0) are already index order (daggers are left/ij)
    # swap i/j k/l so that i>j and k>l
    # now swap the larger one to the left, will cause lots of them to cancel

    n = vijkl_sparse.ndim
    assert n % 2 == 0
    if n > 2:
        ij = vijkl_sparse.coords[: n // 2]
        kl = vijkl_sparse.coords[n // 2 :]
        a = vijkl_sparse.data.copy()

        perm_ij = np.argsort(-ij, axis=0)
        perm_kl = np.argsort(-kl, axis=0)

        ij_desc = ij[perm_ij, np.arange(ij.shape[1])]
        kl_desc = kl[perm_kl, np.arange(kl.shape[1])]

        a *= 1 - 2 * (parity(perm_ij.T) ^ parity(perm_kl.T))
        if set_zero_same:
            # set to zero / remove all those where we try to create / destroy two on the same orbital
            mask = (np.diff(ij_desc, axis=0) == 0).any(axis=0) | (
                np.diff(kl_desc, axis=0) == 0
            ).any(axis=0)
            a = a[~mask]
            ij_desc = ij_desc[:, ~mask]
            kl_desc = kl_desc[:, ~mask]
        new_coords = np.array([*ij_desc, *kl_desc])
        # use coo to merge same indices
        vijkl_sparse = sparse.COO(new_coords, a, shape=vijkl_sparse.shape)
        # we might have some new almost zeros from the cancellations, make sure they are 0
        new_coords2 = vijkl_sparse.coords
        a2 = vijkl_sparse.data
        mask = np.abs(a2) > cutoff
        return sparse.COO(new_coords2[:, mask], a2[mask], shape=vijkl_sparse.shape)
    else:
        return vijkl_sparse


def spinorb_from_spatial_sparse(tij_sparse, vijkl_sparse, interleave=False):
    return spinorb_from_spatial_sparse_coo2(
        tij_sparse, interleave
    ), spinorb_from_spatial_sparse_coo4(vijkl_sparse, interleave)


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


def TV_from_pyscf_molecule(
    molecule,  # type: pyscf.gto.mole.Mole  # noqa: F821
    mo_coeff: np.ndarray,
    *,
    cutoff: float = 1e-11,
) -> tuple[...]:
    r"""
    Computes the nuclear repulsion energy :math:`E_{nuc}`, and the T and
    V tensors encoding the 1-body and 2-body terms in the electronic
    hamiltonian of a pyscf molecule using the specified molecular orbitals.

    The tensors returned correspond to the following expressions:

    .. math::

        \hat{H} = E_{nuc} + \sum_{ij} T_{ij} \hat{c}^\dagger_i\hat{c}_j +
            \sum_{ijkl} V_{ijkl} \hat{c}^\dagger_i\hat{c}^\dagger_j\hat{c}_k\hat{c}_l

    The electronic spin degree of freedom is encoded following the *NetKet convention*
    where the first :math:`N_{\downarrow}` values of the indices :math:`i,j,k,l` represent
    the spin down electrons, and the following :math:`N_{\uparrow}` values represent the
    spin up.

    .. note::

        In the `netket.experimental.operator.pyscf` module you can find some utility
        functions to convert from normal ordering to other orderings, but
        those are all internals so if you need them do copy-paste them somewhere else.

    Example:
        Constructs the hamiltonian for a Li-H molecule, using the `sto-3g` basis
        and the Boys orbitals

        >>> from pyscf import gto, scf, lo
        >>> import netket as nk; import netket.experimental as nkx
        >>>
        >>> geometry = [('Li', (0., 0., -1.5109/2)), ('H', (0., 0., 1.5109/2))]
        >>> mol = gto.M(atom=geometry, basis='STO-3G')
        >>>
        >>> # compute the boys orbitals
        >>> mf = scf.RHF(mol).run(verbose=0)  # doctest:+ELLIPSIS
        >>> mo_coeff = lo.Boys(mol).kernel(mf.mo_coeff)
        >>> ha = nkx.operator.pyscf.TV_from_pyscf_molecule(mol, mo_coeff)

    Args:
        molecule: The pyscf :class:`~pyscf.gto.mole.Mole` object describing the
            Hamiltonian
        mo_coeff: The molecular orbital coefficients determining the
            linear combination of atomic orbitals to produce the
            molecular orbitals. If unspecified this defaults to
            the hartree fock orbitals computed using :class:`~pyscf.scf.HF`.
        cutoff: Ignores all matrix elements in the `V` and `T` matrix that have
            magnitude less than this value. Defaults to :math:`10^{-11}`

    Returns:
        :code:`E,T,V`: a scalar and two numpy arrays, the first with 2 dimensions
        and the latter with 4 dimensions.

    """
    sparse = import_optional_dependency("sparse", descr="TV_from_pyscf_molecule")

    # Compute the T and V matrices
    E_nuc, tij, vijkl = compute_pyscf_integrals(molecule, mo_coeff)
    tij_sparse = sparse.COO.from_numpy(tij * (np.abs(tij) >= cutoff))
    vijkl_sparse = sparse.COO.from_numpy(vijkl * (np.abs(vijkl) >= cutoff))
    tij_sparse_spin, vijkl_sparse_spin = spinorb_from_spatial_sparse(
        tij_sparse, vijkl_sparse
    )
    vijkl_sparse_spin = to_desc_order_sparse(vijkl_sparse_spin, cutoff)

    return E_nuc, tij_sparse_spin, vijkl_sparse_spin
