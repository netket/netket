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

from typing import Optional
from collections.abc import Iterable
import warnings

import numpy as np
from fractions import Fraction

from netket.hilbert.fock import Fock
from netket.hilbert.tensor_hilbert_discrete import TensorDiscreteHilbert
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.utils.dispatch import dispatch


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
        *,
        n_fermions: Optional[int] = None,
        n_fermions_per_spin: Optional[tuple[int, ...]] = None,
    ):
        r"""
        Constructs the hilbert space for spin-`s` fermions on `n_orbitals`.

        Samples of this hilbert space represent occupation numbers :math:`(0,1)` of the
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
            n_fermions: (optional) fixed number of fermions per spin (conserved). If specified,
                the total number of fermions is conserved. If the fermions have a spin, the
                number of fermions per spin subsector is not conserved.
            n_fermions_per_spin: (optional) list of the fixed number of fermions for every spin
                subsector. This automatically enforces a global population constraint. Only one
                between **n_fermions** and **n_fermions_per_spin** can be specified. The length
                of the iterable should be :math:`2S+1`.

        Returns:
            A SpinOrbitalFermions object
        """
        # TODO remove in 3.12 because it's deprecated experimental behaviour
        if isinstance(n_fermions, Iterable):
            warnings.warn(
                """
                          Declaring per-spin subsector constraint with `n_fermions` is deprecated.
                          Use `n_fermions_per_spin` instead.

                          As this class was experimental, this behaviour will break in NetKet 3.12 to be
                          released in early 2024.
                          """,
                DeprecationWarning,
                stacklevel=2,
            )
            n_fermions_per_spin = n_fermions
            n_fermions = None

        if not (isinstance(n_fermions, int) or n_fermions is None):
            raise TypeError(
                """
                The global `n_fermions` constraint must be an integer or be left unspecified.

                To declare a per-spin constraint, specify `n_fermions_per_spin` instead
                """
            )

        if s is None:
            spin_size = 1
        else:
            spin_size = round(2 * s + 1)

        total_size = n_orbitals * spin_size

        if spin_size == 1:
            if n_fermions_per_spin is not None:
                raise ValueError(
                    "Cannot specify `n_fermions_per_spin` for spin-less " "fermions."
                )

            n_fermions_per_spin = (n_fermions,)
            hilbert = Fock(n_max=1, N=n_orbitals, n_particles=n_fermions)
        else:
            if n_fermions_per_spin is not None and n_fermions is not None:
                raise ValueError(
                    """
                                 Cannot specify the per-subsector constraint `n_fermions_per_spin`
                                 at the same time as the global constraint `n_fermions`.

                                 If you want to conserve the total number of fermions, but allow for
                                 changes in the number of fermions with a certain value of spin, you
                                 should specify `n_fermions`. If you want to fix the number of fermions
                                 with each spin component (and therefore the total number of fermions
                                 as well) you should specify `n_fermions_per_spin`.
                                 """
                )
            elif n_fermions_per_spin is not None:
                if not isinstance(n_fermions_per_spin, Iterable):
                    raise TypeError(
                        f"n_fermions_per_spin={n_fermions_per_spin} (whose type is {type(n_fermions_per_spin)}) "
                        "must be a list of integers or None describing the number of "
                        f"fermions in each of the {total_size} spin subsectors."
                    )
                if len(n_fermions_per_spin) != spin_size:
                    raise ValueError(
                        "List of number of fermions must equal number of spin components.\n"
                        f"For s={s}, which has {total_size} components, the same length is "
                        f"expected."
                    )
                n_fermions = sum(n_fermions_per_spin)
                spin_hilberts = [
                    Fock(n_max=1, N=n_orbitals, n_particles=Nf)
                    for Nf in n_fermions_per_spin
                ]
                hilbert = TensorDiscreteHilbert(*spin_hilberts)
            else:
                # fixed fermion number but multiple spins subspaces
                # global or no constraint
                n_fermions_per_spin = tuple(None for _ in range(spin_size))
                hilbert = Fock(n_max=1, N=total_size, n_particles=n_fermions)

        self._fock = hilbert
        """Internal representation of this Hilbert space (Fock or TensorHilbert)."""
        # local states are the occupation numbers (0, 1)
        local_states = np.array((0.0, 1.0))

        # we use the constraints from the Fock spaces, and override `constrained`
        super().__init__(local_states, N=total_size, constraint_fn=None)
        self._s = s
        self._n_fermions = n_fermions
        self._n_fermions_per_subsector: tuple[Optional[int], ...] = n_fermions_per_spin
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
        r"""The hilbert space does not contains `prod(hilbert.shape)`
        basis states.

        Typical constraints are population constraints (such as fixed
        number of bosons, fixed magnetization...) which ensure that
        only a subset of the total unconstrained space is populated.
        """
        return self._n_fermions is not None

    @property
    def is_finite(self) -> bool:
        """True if the hilbert space can be indexed by a single
        32-bit unsigned integer.
        """
        return self._fock.is_finite

    @property
    def n_states(self) -> int:
        """
        Total size of the hilbert space, if indexable.
        """
        return self._fock.n_states

    @property
    def _n_spin_states(self) -> int:
        """Return the number of spin projections"""
        if self.spin is None:
            raise Exception(
                "cannot request number of spin states for spinless fermions"
            )
        return round(2 * self.spin + 1)

    def _spin_index(self, sz: Optional[int]) -> int:
        """Return the index of the Fock block corresponding to the sz projection"""
        if self.spin is None:
            if sz is not None or not np.isclose(sz, 0):
                raise Exception("cannot request spin index of spinless fermions")
            return 0
        else:
            if sz != int(sz):
                raise TypeError(f"sz must be an integer, but got {type(sz)} ({sz})")
            if abs(sz) > 2 * self.spin:
                val = int(2 * self.spin)
                raise ValueError(
                    f"Valid spin values are in the interval [{-val},{val}]"
                )
            # if integer spin, valid values are even
            if int(self.spin) == self.spin:
                if sz % 2 != 0:
                    raise ValueError(
                        f"For spin S={self.spin}, valid spin values are odd"
                    )
            else:
                # valid values are odd
                if sz % 2 == 0:
                    raise ValueError(
                        f"For spin S={self.spin}, valid spin values are even"
                    )
            return (sz + int(2 * self.spin)) // 2

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
        if self.spin is not None:
            _str += f", s={Fraction(self.spin)}"
        if self.n_fermions is not None:
            _str += f", n_fermions={self.n_fermions}"
        if self.spin is not None and any(
            x is not None for x in self.n_fermions_per_spin
        ):
            _str += f", n_fermions_per_spin={self.n_fermions_per_spin}"
        _str += ")"
        return _str


@dispatch
def random_state(hilb: SpinOrbitalFermions, key, batches: int, *, dtype):
    return random_state(hilb._fock, key, batches, dtype)


@dispatch
def flip_state_scalar(hilb: SpinOrbitalFermions, key, state, index):
    return flip_state_scalar(hilb._fock, key, state, index)


@dispatch
def flip_state_batch(hilb: SpinOrbitalFermions, key, state, index):
    return flip_state_batch(hilb._fock, key, state, index)
