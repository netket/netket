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


import numpy as np


from netket.operator import DiscreteOperator
from netket.hilbert import SpinOrbitalFermions
from netket.utils.optional_deps import import_optional_dependency
from netket.operator import FermionOperator2nd, FermionOperator2ndJax
from ._particle_number_conserving_fermionic import (
    ParticleNumberConservingFermioperator2nd,
    ParticleNumberAndSpinConservingFermioperator2nd,
)

from ._pyscf_utils import arrays_to_terms, TV_from_pyscf_molecule


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


def from_pyscf_molecule(
    molecule,  # type: pyscf.gto.mole.Mole  # noqa: F821
    mo_coeff: np.ndarray | None = None,
    *,
    cutoff: float = 1e-11,
    implementation: DiscreteOperator = ParticleNumberAndSpinConservingFermioperator2nd,
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
        >>> mf = scf.RHF(mol).run(verbose=0)  # doctest:+ELLIPSIS
        >>> E_hf = sum(mf.scf_summary.values())
        >>>
        >>> E_fci = fci.FCI(mf).kernel()[0]
        >>>
        >>> ha = nkx.operator.from_pyscf_molecule(mol)  # doctest:+ELLIPSIS
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
        >>> mf = scf.RHF(mol).run(verbose=0)  # doctest:+ELLIPSIS
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
            :class:`netket.experimental.operator.ParticleNumberAndSpinConservingFermioperator2nd` (this might
            change in the future).

    Returns:
        A netket second quantised operator that encodes the electronic hamiltonian.
    """
    # If unspecified, use HF molecular orbitals
    if mo_coeff is None:
        pyscf = import_optional_dependency("pyscf", descr="pyscf_molecule_to_arrays")

        mf = pyscf.scf.HF(molecule).run(verbose=0)
        mo_coeff = mf.mo_coeff

    if implementation in [FermionOperator2ndJax, FermionOperator2ndJax]:
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
    elif implementation in [
        ParticleNumberConservingFermioperator2nd,
        ParticleNumberAndSpinConservingFermioperator2nd,
    ]:
        return implementation.from_pyscf_molecule(molecule, mo_coeff)
    else:
        raise ValueError("unknown implementation")
