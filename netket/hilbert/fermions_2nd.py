from typing import Optional, List
from netket.hilbert.fock import Fock
from collections.abc import Iterable
from netket.hilbert.homogeneous import HomogeneousHilbert


class SpinOrbitalFermions(HomogeneousHilbert):
    def __init__(
        self,
        n_orbitals: int,
        n_fermions: Optional[int] = None,
        n_fermions_per_spin: Optional[List[int]] = None,
    ):
        r"""
        Helper function generating fermions in 2nd quantization with n_orbitals.
        Samples of this hilbert space represents occupations numbers (0,1) of the orbitals.
        Number of fermions can be fixed by setting n_fermions.
        Using this class, one can generate a tensor product of fermionic hilbert spaces that distinguish particles with different spin.
        Internally, this
        Args:
            n_orbitals (required): number of orbitals we store occupation numbers for. If the number of fermions per spin is converved, the different spin configurations are not counted as orbitals and are handled differently.
            n_fermions (optional, int): fixed number of fermions that occupy the orbitals.
            n_fermions_per_spin (optional, list(int)): fixed number of fermions per spin (conserved). If not None, the returned object will be a tensorproduct of Fock spaces.
        Returns:
            A SpinOrbitalFermions object
        """
        # checks
        if n_fermions_per_spin is not None:
            if not isinstance(n_fermions_per_spin, Iterable):
                raise ValueError("n_fermions_per_spin must be iterable")
            if n_fermions is not None and sum(n_fermions_per_spin) != int(n_fermions):
                raise ValueError("n_fermions and n_fermions_per_spin do not agree")
            hilbert = None
            for Nf in n_fermions_per_spin:
                spin_hilbert = Fock(1, N=n_orbitals, n_particles=Nf)
                if hilbert is None:
                    hilbert = spin_hilbert
                else:
                    hilbert *= spin_hilbert
        else:
            hilbert = Fock(n_max=1, N=n_orbitals, n_particles=n_fermions)
        return hilbert
