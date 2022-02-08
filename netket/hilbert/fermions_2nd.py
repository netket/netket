from re import L
from typing import Optional, List, Union
from xml.sax.handler import property_declaration_handler
from netket.hilbert.fock import Fock
from collections.abc import Iterable
from netket.hilbert.homogeneous import HomogeneousHilbert
import jax.numpy as jnp
from netket.hilbert.tensor_hilbert import TensorHilbert
import numpy as np


class OrbitalFermions(HomogeneousHilbert):
    def __init__(self, n_orbitals: int, n_fermions: Optional[int] = None):
        r"""
        Helper function generating fermions in 2nd quantization with n_orbitals.
        Samples of this hilbert space represents occupations numbers (0,1) of the orbitals.
        Number of fermions can be fixed by setting n_fermions.
        Args:
            n_orbitals (required): number of orbitals we store occupation numbers for. If the number of fermions per spin is converved, the different spin configurations are not counted as orbitals and are handled differently.
            n_fermions (optional, int): fixed number of fermions that occupy the orbitals.
        Returns:
            A OrbitalFermions object
        """
        self._fock = Fock(n_max=1, N=n_orbitals, n_particles=n_fermions)
        local_states = np.array((0.0, 1.0))
        super().__init__(local_states, N=n_orbitals, constraint_fn=None)
        self.ptrace = self._fock.ptrace
        self._numbers_to_states = self._fock._numbers_to_states
        self._states_to_numbers = self._fock._states_to_numbers
        self.__mul__ = self._fock.__mul__

    @property
    def n_orbitals(self):
        return self._fock.size

    @property
    def n_fermions(self):
        return self._fock.n_particles

    def __repr__(self):
        _str = "OrbitalFermions(n_orbitals={}".format(self.n_orbitals)
        if self.n_fermions is not None:
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
    def n_spin_states(self) -> int:
        return 1

    @property
    def spin(self):
        return 0.0

    def spin_index(self, sz: float) -> int:
        return 0

    def get_index(self, site: int, sz: float = None):
        if not sz is None or sz == 0.0:
            raise ValueError("OrbitalFermions have sz=0")
        return site


class SpinOrbitalFermions(HomogeneousHilbert):
    def __init__(
        self, n_orbitals: int, s: float = 0.0, n_fermions: Union[int, List[int]] = None
    ):
        r"""
        Helper function generating fermions in 2nd quantization with n_orbitals.
        Samples of this hilbert space represents occupations numbers (0,1) of the orbitals.
        Number of fermions can be fixed by setting n_fermions.
        Using this class, one can generate a tensor product of fermionic hilbert spaces that distinguish particles with different spin.
        Internally, this
        Args:
            n_orbitals (required): number of orbitals we store occupation numbers for. If the number of fermions per spin is converved, the different spin configurations are not counted as orbitals and are handled differently.
            n_fermions_per_spin (optional, list(int)): fixed number of fermions per spin (conserved). If not None, the returned object will be a tensorproduct of Fock spaces.
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
            hilbert = OrbitalFermions(total_size)
        else:
            if isinstance(n_fermions, int):
                hilbert = OrbitalFermions(total_size, n_fermions=n_fermions)
            else:
                if not isinstance(n_fermions, Iterable):
                    raise ValueError("n_fermions must be iterable or int")
                if len(n_fermions) != spin_size:
                    raise ValueError(
                        "list of number of fermions must equal number of spin components"
                    )
                spin_hilberts = [OrbitalFermions(n_orbitals, Nf) for Nf in n_fermions]
                hilbert = TensorHilbert(*spin_hilberts)

        # internally, we store a Fock space or a TensorHilbert of Fock spaces
        self._fock = hilbert
        local_states = np.array((0.0, 1.0))

        super().__init__(local_states, N=total_size, constraint_fn=None)
        self._s = s
        self.n_fermions = n_fermions
        self.n_orbitals = n_orbitals
        self._n_spin_components = spin_size
        self._spin_states = spin_states
        # we copy the respective functions, independent of what hilbert space they are
        self.ptrace = self._fock.ptrace
        self._numbers_to_states = self._fock._numbers_to_states
        self._states_to_numbers = self._fock._states_to_numbers
        self.__mul__ = self._fock.__mul__

    def __repr__(self):
        _str = "SpinOrbitalFermions(n_orbitals={}".format(self.n_orbitals)
        if self.n_fermions is not None:
            _str += ", n_fermions={}".format(self.n_fermions)
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
        return self._fock._attrs

    @property
    def is_finite(self) -> bool:
        return self._fock.is_finite

    @property
    def n_states(self) -> int:
        return self._fock.n_states

    @property
    def n_spin_states(self) -> int:
        return self._n_spin_components

    @property
    def spin_states(self) -> List[float]:
        return tuple(self._spin_states)

    def spin_index(self, sz: float) -> int:
        return round(2 * sz + 2 * self.spin)

    def get_index(self, site: int, sz: float = None):
        """go from (site, spin_projection) indices to index in the (tensor) hilbert space"""
        n_spin_states = self.n_spin_states
        spin_idx = self.spin_index(sz)
        return spin_idx * n_spin_states + site
