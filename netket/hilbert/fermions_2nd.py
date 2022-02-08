from typing import Optional, List
from netket.hilbert.fock import Fock
from collections.abc import Iterable
from netket.hilbert.homogeneous import HomogeneousHilbert
import jax.numpy as jnp


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
            n_fermions = sum(n_fermions_per_spin)
            hilbert = None
            for Nf in n_fermions_per_spin:
                spin_hilbert = Fock(1, N=n_orbitals, n_particles=Nf)
                if hilbert is None:
                    hilbert = spin_hilbert
                    local_states = hilbert.local_states
                else:
                    hilbert *= spin_hilbert
            _sizes = (n_orbitals,) * len(n_fermions_per_spin)
        else:
            hilbert = Fock(n_max=1, N=n_orbitals, n_particles=n_fermions)
            local_states = hilbert.local_states
            _sizes = (n_orbitals,)

        # internally, we store a Fock space or a TensorHilbert of Fock spaces
        self._fock = hilbert
        super().__init__(local_states, N=n_orbitals, constraint_fn=None)
        self._n_orbitals = n_orbitals
        self._n_fermions = n_fermions
        self._n_fermions_per_spin = n_fermions_per_spin
        self._sizes = _sizes
        # we copy the respective functions, independent of what hilbert space they are
        self.ptrace = self._fock.ptrace
        self._numbers_to_states = self._fock._numbers_to_states
        self._states_to_numbers = self._fock._states_to_numbers
        self.__mul__ = self._fock.__mul__

    def __repr__(self):
        _str = "SpinOrbitalFermions(n_orbitals={}".format(self.n_orbitals)
        if self._n_fermions_per_spin is not None:
            _str += ", n_fermions_per_spin={}".format(self._n_fermions_per_spin)
        elif self._n_fermions is not None:
            _str += ", n_fermions={}".format(self._n_fermions)
        _str += ")"
        return _str

    @property
    def size(self) -> int:
        return self._fock.size

    @property
    def _attrs(self):
        return self._fock._attrs

    @property
    def is_finite(self) -> bool:
        return self._fock.is_finite

    @property
    def n_states(self) -> int:
        return self._fock.n_states

    @property
    def n_orbitals(self) -> int:
        return self._n_orbitals

    @property
    def n_spin_components(self) -> int:
        return len(self._sizes)
