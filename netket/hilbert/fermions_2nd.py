from re import L
from typing import Optional, List, Union
from xml.sax.handler import property_declaration_handler
from netket.hilbert.fock import Fock
from collections.abc import Iterable
from netket.hilbert.homogeneous import HomogeneousHilbert
import jax.numpy as jnp
from netket.hilbert.tensor_hilbert import TensorHilbert
import numpy as np
from fractions import Fraction


class SpinOrbitalFermions(HomogeneousHilbert):
    def __init__(
        self, n_orbitals: int, s: float = 0.0, n_fermions: Union[int, List[int]] = None
    ):
        r"""
        Class of fermions in 2nd quantization with n_orbitals and spin s.
        Samples of this hilbert space represent occupation numbers (0,1) of the orbitals.
        Number of fermions can be fixed by setting n_fermions.
        If the spin is different from 0, n_fermions can also be a list to fix the number of fermions per spin component.
        Using this class, one can generate a tensor product of fermionic hilbert spaces that distinguish particles with different spin.

        Args:
            n_orbitals (required): number of orbitals we store occupation numbers for. If the number of fermions per spin is conserved, the different spin configurations are not counted as orbitals and are handled differently.
            s (float): spin of the fermions.
            n_fermions (optional, int or list(int)): fixed number of fermions per spin (conserved). In the case n_fermions is an int, the total number of fermions is fixed, while for lists, the number of fermions per spin component is fixed.

        Returns:
            A SpinOrbitalFermions object
        """
        spin_size = round(2 * s + 1)
        spin_states = np.empty(spin_size)
        for i in range(spin_size):
            spin_states[i] = 2 * i - round(2 * s)
        spin_states = spin_states.tolist()

        total_size = n_orbitals * spin_size
        if n_fermions is None:
            hilbert = Fock(n_max=1, N=total_size)
        else:
            if isinstance(n_fermions, int):
                hilbert = Fock(n_max=1, N=total_size, n_particles=n_fermions)
            else:
                if not isinstance(n_fermions, Iterable):
                    raise ValueError("n_fermions must be iterable or int")
                if len(n_fermions) != spin_size:
                    raise ValueError(
                        "list of number of fermions must equal number of spin components"
                    )
                spin_hilberts = [
                    Fock(n_max=1, N=n_orbitals, n_particles=Nf) for Nf in n_fermions
                ]
                hilbert = TensorHilbert(*spin_hilberts)

        # internally, we store a Fock space or a TensorHilbert of Fock spaces
        self._fock = hilbert
        local_states = np.array((0.0, 1.0))

        super().__init__(local_states, N=total_size, constraint_fn=None)
        self._s = s
        self.n_fermions = n_fermions
        self.n_orbitals = n_orbitals
        self._n_spin_states = spin_size
        self._spin_states = tuple(spin_states)
        # we copy the respective functions, independent of what hilbert space they are
        self._numbers_to_states = self._fock._numbers_to_states
        self._states_to_numbers = self._fock._states_to_numbers

    def __repr__(self):
        _str = "SpinOrbitalFermions(n_orbitals={}".format(self.n_orbitals)
        if self.n_fermions is not None:
            _str += ", n_fermions={}".format(self.n_fermions)
        if self.spin != 0.0:
            _str += ", s={}".format(Fraction(self.spin))
        _str += ")"
        return _str

    @property
    def spin(self) -> float:
        return self._s

    @property
    def size(self) -> int:
        return self._fock.size

    @property
    def _attrs(self):
        return (
            self.spin,
            self.n_fermions,
            self.n_orbitals,
            self._n_spin_states,
            self._spin_states,
        )

    @property
    def is_finite(self) -> bool:
        return self._fock.is_finite

    @property
    def n_states(self) -> int:
        return self._fock.n_states

    def _spin_index(self, sz: float) -> int:
        """return the index of the Fock block corresponding to the sz projection"""
        return round(sz + self.spin)

    def _get_index(self, orb: int, sz: float = None):
        """go from (site, spin_projection) indices to index in the (tensor) hilbert space"""
        spin_idx = self._spin_index(sz)
        return spin_idx * self.n_orbitals + orb
