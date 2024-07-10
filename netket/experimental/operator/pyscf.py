# Copyright 2023 The NetKet Authors - All rights reserved.
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

from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp

from netket.operator import DiscreteOperator
from netket.experimental.hilbert import SpinOrbitalFermions
from netket.utils.optional_deps import import_optional_dependency
from ._fermion_operator_2nd_numba import FermionOperator2nd


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
    sparse = import_optional_dependency("sparse", descr="TV_from_pyscf_molecule")

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
    sparse = import_optional_dependency("sparse", descr="TV_from_pyscf_molecule")

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
    sparse = import_optional_dependency("sparse", descr="TV_from_pyscf_molecule")

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
                n_orbitals=tij_sparse.shape[0] // 2,
                s=1 / 2,
                n_fermions_per_spin=n_electrons,
            )
        else:
            hi = SpinOrbitalFermions(
                n_orbitals=tij_sparse.shape[0], n_fermions_per_spin=n_electrons
            )
    terms, weights, constant = arrays_to_terms(
        const, tij_sparse, vijkl_sparse, term_conj2=term_conj2, term_conj4=term_conj4
    )
    return cls(hi, terms, weights, constant)


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
            \sum_{ijkl} V_{ijkl} \hat{c}^\dagger_i\hat{c}_\dagger_j\hat{c}_k\hat{c}_l

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
        >>> mf = scf.RHF(mol).run()  # doctest:+ELLIPSIS
            converged SCF energy = -7.86338...
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


def from_pyscf_molecule(
    molecule,  # type: pyscf.gto.mole.Mole  # noqa: F821
    mo_coeff: Optional[np.ndarray] = None,
    *,
    cutoff: float = 1e-11,
    implementation: DiscreteOperator = FermionOperator2nd,
) -> DiscreteOperator:
    r"""
    Construct a netket operator encoding the electronic hamiltonian of a pyscf
    molecule in a chosen orbital basis.

    Example:
        Constructs the hamiltonian for a Li-H molecule, using the `sto-3g` basis
        and the default Hartree-Fock molecular orbitals.

        >>> from pyscf import gto, scf, fci
        >>> import netket as nk; import netket.experimental as nkx
        >>>
        >>> bond_length = 1.5109
        >>> geometry = [('Li', (0., 0., -bond_length/2)), ('H', (0., 0., bond_length/2))]
        >>> mol = gto.M(atom=geometry, basis='STO-3G')
        >>>
        >>> mf = scf.RHF(mol).run()  # doctest:+ELLIPSIS
            converged SCF energy = -7.86338...
        >>> E_hf = sum(mf.scf_summary.values())
        >>>
        >>> E_fci = fci.FCI(mf).kernel()[0]
        >>>
        >>> ha = nkx.operator.from_pyscf_molecule(mol)  # doctest:+ELLIPSIS
            converged SCF energy = -7.86338...
        >>> E0 = float(nk.exact.lanczos_ed(ha))
        >>> print(f"{E0 = :.5f}, {E_fci = :.5f}")
        E0 = -7.88253, E_fci = -7.88253

    Example:
        Constructs the hamiltonian for a Li-H molecule, using the `sto-3g` basis
        and the Boys orbitals using :class:`~pyscf.lo.Boys`.

        >>> from pyscf import gto, scf, lo
        >>> import netket as nk; import netket.experimental as nkx
        >>>
        >>> bond_length = 1.5109
        >>> geometry = [('Li', (0., 0., -bond_length/2)), ('H', (0., 0., bond_length/2))]
        >>> mol = gto.M(atom=geometry, basis='STO-3G')
        >>>
        >>> # compute the boys orbitals
        >>> mf = scf.RHF(mol).run()  # doctest:+ELLIPSIS
            converged SCF energy = -7.86338...
        >>> mo_coeff = lo.Boys(mol).kernel(mf.mo_coeff)
        >>> # use the boys orbitals to construct the netket hamiltonian
        >>> ha = nkx.operator.from_pyscf_molecule(mol, mo_coeff=mo_coeff)

    Args:
        molecule: The pyscf :class:`~pyscf.gto.mole.Mole` object describing the
            Hamiltonian
        mo_coeff: The molecular orbital coefficients determining the
            linear combination of atomic orbitals to produce the
            molecular orbitals. If unspecified this defaults to
            the hartree fock orbitals computed using :class:`~pyscf.scf.HF`.
        cutoff: Ignores all matrix elements in the `V` and `T` matrix that have
            magnitude less than this value. Defaults to :math:`10^{-11}`
        implementation: The particular implementation to use for the operator.
            Different fermionic operator implementation might have different
            performances. Defaults to
            :class:`netket.experimental.operator.FermionOperator2nd` (this might
            change in the future).

    Returns:
        A netket second quantised operator that encodes the electronic hamiltonian.
    """
    # If unspecified, use HF molecular orbitals
    if mo_coeff is None:
        pyscf = import_optional_dependency("pyscf", descr="pyscf_molecule_to_arrays")

        mf = pyscf.scf.HF(molecule).run()
        mo_coeff = mf.mo_coeff

    E_nuc, Tij, Vijkl = TV_from_pyscf_molecule(molecule, mo_coeff, cutoff=cutoff)

    ha = operator_from_arrays(
        E_nuc,
        Tij,
        0.5 * Vijkl,
        molecule.nelec,
        term_conj4=(1, 1, 0, 0),
        cls=implementation,
    )
    # TODO maybe run setup and set _max_conn_size here estimating it analytially
    return ha
