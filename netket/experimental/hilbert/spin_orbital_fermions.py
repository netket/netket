# Copyright 2022-2023 The NetKet Authors - All rights reserved.
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

from typing import Optional, Union
from collections.abc import Iterable
import numpy as np
from fractions import Fraction

from netket.hilbert.fock import Fock
from netket.hilbert.tensor_hilbert_discrete import TensorDiscreteHilbert
from netket.hilbert.homogeneous import HomogeneousHilbert


class SpinOrbitalFermions(HomogeneousHilbert):
    r"""
    Hilbert space for 2nd quantization fermions with spin `s` distributed among
    `n_orbital` orbitals.

    The number of fermions can be fixed globally or fixed on a per spin projection.

    Note:
        This class is simply a convenient wrapper that creates a Fock or TensorHilbert
        of Fock spaces with occupation numbers 0 or 1.
        It is mainly useful to avoid needing to specify the n_max=1 each time, and adds
        convenient functions such as _get_index and _spin_index, which allow one to
        index the correct TensorHilbert corresponding to the right spin projection.
    """

    def __init__(
        self,
        n_orbitals: int,
        s: Optional[float] = None,
        n_fermions: Optional[Union[int, list[int]]] = None,
    ):
        r"""
        Constructs the hilbert space for spin-`s` fermions on `n_orbitals`.

        Samples of this hilbert space represent occupation numbers (0,1) of the
        orbitals. The number of fermions may be fixed to `n_fermions`.
        If the spin is different from 0 or None, n_fermions can also be a list to fix
        the number of fermions per spin component.
        Using this class, one can generate a tensor product of fermionic hilbert spaces
        that distinguish particles with different spin.

        Args:
            n_orbitals: number of orbitals we store occupation numbers for. If the
                number of fermions per spin is conserved, the different spin
                configurations are not counted as orbitals and are handled differently.
            s: spin of the fermions.
            n_fermions: (optional) fixed number of fermions per spin (conserved). In the
                case n_fermions is an int, the total number of fermions is fixed, while
                for lists, the number of fermions per spin component is fixed.

        Returns:
            A SpinOrbitalFermions object
        """
        if s is None:
            spin_size = 1
        else:
            spin_size = round(2 * s + 1)

        total_size = n_orbitals * spin_size

        if spin_size == 1:
            if not isinstance(n_fermions, Optional[int]):
                raise TypeError(
                    "Spinless fermions require that `n_fermions` be "
                    f"an integer or None. (Got {n_fermions})"
                )
            hilbert = Fock(n_max=1, N=n_orbitals, n_particles=n_fermions)
            n_fermions_s = (n_fermions,)
        elif isinstance(n_fermions, int):
            # fixed fermion number but multiple spins subspaces
            hilbert = Fock(n_max=1, N=total_size, n_particles=n_fermions)
            n_fermions_s = tuple(None for _ in range(spin_size))
        else:
            if n_fermions is None:
                n_fermions_s = tuple(None for _ in range(spin_size))
            else:
                if not isinstance(n_fermions, Iterable):
                    raise TypeError(
                        f"n_fermions={n_fermions} (whose type is {type(n_fermions)}) "
                        "must be None or a list of integers describing the number of "
                        f"fermions in each of the {total_size} spin subsectors."
                    )
                n_fermions_s = n_fermions
                n_fermions = sum(n_fermions)

            if len(n_fermions_s) != spin_size:
                raise ValueError(
                    "List of number of fermions must equal number of spin components.\n"
                    f"For s={s}, which has {total_size} components, the same length is "
                    f"expected."
                )

            spin_hilberts = [
                Fock(n_max=1, N=n_orbitals, n_particles=Nf) for Nf in n_fermions_s
            ]
            hilbert = TensorDiscreteHilbert(*spin_hilberts)

        self._fock = hilbert
        """Internal representation of this Hilbert space (Fock or TensorHilbert)."""
        # local states are the occupation numbers (0, 1)
        local_states = np.array((0.0, 1.0))

        # we use the constraints from the Fock spaces, and override `constrained`
        super().__init__(local_states, N=total_size, constraint_fn=None)
        self._s = s
        self._n_fermions = n_fermions
        self._n_fermions_per_subsector: tuple[Optional[int], ...] = n_fermions_s
        self._n_orbitals = n_orbitals

        # we copy the respective functions, independent of what hilbert space they are
        self._numbers_to_states = self._fock._numbers_to_states
        self._states_to_numbers = self._fock._states_to_numbers
        self.all_states = self._fock.all_states

    @property
    def n_fermions(self) -> Optional[int]:
        """The total number of fermions. None if unspecified."""
        return self._n_fermions

    @property
    def n_fermions_per_spin(self) -> tuple[Optional[int], ...]:
        """Tuple identifying the per-subsector population constraint.

        This tuple has length 1 for spinless fermions and length 2s+1 for spinful fermions.
        Every element is an integer or None, where None means no constraint on the number
        of fermions in that subsector.
        """
        return self._n_fermions_per_subsector

    @property
    def n_spin_subsectors(self) -> int:
        """Total number of spin subsectors. If spin is None, this is 1."""
        return len(self._n_fermions_per_subsector)

    @property
    def n_orbitals(self) -> int:
        """The number of orbitals (for each spin subsector if relevant)."""
        return self._n_orbitals

    @property
    def spin(self) -> float:
        """Returns the spin of the fermions"""
        return self._s

    @property
    def size(self) -> int:
        """Size of the hilbert space. In case the fermions have spin `s`, the size is
        (2*s+1)*n_orbitals"""
        return self._fock.size

    @property
    def _attrs(self):
        return (self.spin, self.n_fermions, self.n_orbitals)

    @property
    def constrained(self):
        return self._n_fermions is not None

    @property
    def is_finite(self) -> bool:
        return self._fock.is_finite

    @property
    def n_states(self) -> int:
        return self._fock.n_states

    @property
    def _n_spin_states(self) -> int:
        """return the number of spin projections"""
        if self.spin is None:
            raise Exception(
                "cannot request number of spin states for spinless fermions"
            )
        return round(2 * self.spin + 1)

    def _spin_index(self, sz: float) -> int:
        """return the index of the Fock block corresponding to the sz projection"""
        if self.spin is None:
            if sz is not None or not np.isclose(sz, 0):
                raise Exception("cannot request spin index of spinless fermions")
            return 0
        else:
            return round(sz + self.spin)

    def states_to_local_indices(self, x):
        return self._fock.states_to_local_indices(x)

    def _get_index(self, orb: int, sz: Optional[float] = None):
        """go from (site, spin_projection) indices to index in the hilbert space"""
        if orb >= self.n_orbitals:
            raise IndexError("requested orbital index outside of the hilbert space")
        spin_idx = self._spin_index(sz)
        return spin_idx * self.n_orbitals + orb

    def __repr__(self):
        _str = f"SpinOrbitalFermions(n_orbitals={self.n_orbitals}"
        if self.n_fermions is not None:
            _str += f", n_fermions={self.n_fermions}"
        if self.spin is not None:
            _str += f", s={Fraction(self.spin)}"
        _str += ")"
        return _str
