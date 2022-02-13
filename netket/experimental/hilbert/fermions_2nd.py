# Copyright 2022 The NetKet Authors - All rights reserved.
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

from typing import Optional, List, Union
from collections.abc import Iterable
import numpy as np
from fractions import Fraction

from netket.hilbert.fock import Fock
from netket.hilbert.tensor_hilbert import TensorHilbert
from netket.hilbert.homogeneous import HomogeneousHilbert


class SpinOrbitalFermions(HomogeneousHilbert):
    r"""
    Hilbert space for 2nd quantization fermions with spin `s` distributed among `n_orbital` orbitals.

    The number of fermions can be fixed globally or fixed on a per spin projection.
    """

    def __init__(
        self,
        n_orbitals: int,
        s: float = 0.0,
        n_fermions: Optional[Union[int, List[int]]] = None,
    ):
        r"""
        Constructs the hilbert space for spin-`s` fermions on `n_orbitals`.

        Samples of this hilbert space represent occupation numbers (0,1) of the orbitals.
        The number of fermions may be fixed to `n_fermions`.
        If the spin is different from 0, n_fermions can also be a list to fix the number of fermions per spin component.
        Using this class, one can generate a tensor product of fermionic hilbert spaces that distinguish particles with different spin.

        Args:
            n_orbitals: number of orbitals we store occupation numbers for. If the number of fermions per spin is conserved, the different spin configurations are not counted as orbitals and are handled differently.
            s: spin of the fermions.
            n_fermions: (optional) fixed number of fermions per spin (conserved). In the case n_fermions is an int, the total number of fermions is fixed, while for lists, the number of fermions per spin component is fixed.

        Returns:
            A SpinOrbitalFermions object
        """

        spin_size = round(2 * s + 1)
        spin_states = list(np.arange(spin_size) * 2 - round(2 * s))

        total_size = n_orbitals * spin_size
        if n_fermions is None:
            hilbert = Fock(n_max=1, N=total_size)
        elif isinstance(n_fermions, int):
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

        self._fock = hilbert
        """Internal representation of this Hilbert space, which is either a Fock or TensorHilbert."""
        # local states are the occupation numbers (0, 1)
        local_states = np.array((0.0, 1.0))

        # we use the constraints from the Fock spaces, and override is_constrained later
        super().__init__(local_states, N=total_size, constraint_fn=None)
        self._s = s
        self.n_fermions = n_fermions
        self._is_constrained = n_fermions is not None
        self.n_orbitals = n_orbitals
        self._n_spin_states = spin_size
        self._spin_states = tuple(spin_states)
        # we copy the respective functions, independent of what hilbert space they are
        self._numbers_to_states = self._fock._numbers_to_states
        self._states_to_numbers = self._fock._states_to_numbers

    def __repr__(self):
        _str = f"SpinOrbitalFermions(n_orbitals={self.n_orbitals}"
        if self.n_fermions is not None:
            _str += f", n_fermions={self.n_fermions}"
        if self.spin != 0.0:
            _str += f", s={Fraction(self.spin)}"
        _str += ")"
        return _str

    @property
    def spin(self) -> float:
        """Returns the spin of the fermions"""
        return self._s

    @property
    def size(self) -> int:
        """Size of the hilbert space. In case the fermions have spin `s`, the size is (2*s+1)*n_orbitals"""
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
    def is_constrained(self):
        return self._is_constrained

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
