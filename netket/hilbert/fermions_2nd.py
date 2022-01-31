from typing import Optional, List, Callable

from netket.graph import AbstractGraph

from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.hilbert._deprecations import graph_to_N_depwarn

from collections.abc import Iterable

import numpy as np

class Fermions2nd(HomogeneousHilbert):
    def __init__(
        self,
        n_orbitals: int,
        n_fermions: Optional[int] = None,
        extra_constraint_fn: Optional[Callable] = None,
    ):
        r"""
        Class representing fermions in 2nd quantization with n_orbitals.
        Samples of this hilbert space represents occupations numbers (0,1) of the orbitals.
        Number of fermions can be fixed by setting n_fermions.
        Implementation is optimized for fixed number of fermions.
        Args:
            n_orbitals (required): number of orbitals we store occupation numbers for.
            n_fermions (optional, int): number of fermions that occupy the orbitals.
            extra_constraint_fn (optional, Callable): any additional constraint function that does not include the fermion number constraint.
        Returns:
            A Fermions2nd object.
        """
        local_states = [0, 1]  # occupied or not

        number_constraint_fn = None
        if n_fermions is not None:
            number_constraint_fn = lambda x: np.sum(x, axis=-1) == n_fermions
        else:
            number_constraint_fn = None

        if extra_constraint_fn is not None:
            self._extra_constrained = True
            if number_constraint_fn is None:
                constraint_fn = extra_constraint_fn
            else:
                constraint_fn = lambda x: np.logical_and(
                    extra_constraint_fn(x), number_constraint_fn(x)
                )
        else:
            self._extra_constrained = False
            constraint_fn = number_constraint_fn

        super().__init__(local_states, N=n_orbitals, constraint_fn=constraint_fn)
        self._n_orbitals = n_orbitals
        self._n_fermions = n_fermions

    @property
    def number_constrained(self):
        return self._n_fermions is not None

    @property
    def extra_constrained(self):
        return self._extra_constrained

    @property
    def n_orbitals(self):
        return self._n_orbitals

    @property
    def n_fermions(self):
        return self._n_fermions

    @property
    def n_holes(self):
        return (
            None if (self.n_fermions is None) else (self.n_orbitals - self.n_fermions)
        )

    def __repr__(self):
        return "Fermions2nd(n_orbitals={})".format(self.n_orbitals)


class LatticeFermions2nd(Fermions2nd):
    def __init__(
        self,
        n_sites: int,
        s: Optional[int] = 0,
        n_fermions: Optional[int] = None,
        n_fermions_per_spin: Optional[List[int]] = None,
        graph: Optional[AbstractGraph] = None,
    ):
        r"""
        Class representing fermions in 2nd quantization on a lattice.
        This class is similar to Fermions2nd, but splits off spatial and spin degrees of freedom.
        Args:
            n_sites (required): number of sites in the lattice.
            s (optional, int, default=0): spins of the fermions
            n_fermions_per_spin (optional, int): for each spin projection quantum number, the total (fixed) number of fermions present
            graph (optional, AbstractGraph): an optional graph argument from which we can infer the number of sites
        Returns:
            A LatticeFermions2nd object.
        """
        self._s = s
        self._n_sites = graph_to_N_depwarn(N=n_sites, graph=graph)
        spin_degrees = int(2 * s + 1)
        n_orbitals = spin_degrees * n_sites
        self._spin_sections = (np.arange(spin_degrees) + 1)[:-1] * n_sites

        if n_fermions_per_spin is not None:
            if not isinstance(n_fermions_per_spin, Iterable):
                raise ValueError("n_fermions_per_spin must be iterable")

        if (n_fermions is not None) and (n_fermions_per_spin is not None):
            if not n_fermions == np.sum(n_fermions_per_spin):
                raise ValueError("n_fermions and n_fermions_per_spin do not agree")

        constraint_fn = None

        if n_fermions_per_spin is not None:
            n_fermions_per_spin = np.array(n_fermions_per_spin)
            n_fermions = np.sum(n_fermions_per_spin)
            if not len(n_fermions_per_spin) == spin_degrees:
                raise ValueError("n_fermions_per_spin and s do not agree")

            def constraint_fn(x):
                return np.isclose(
                    self.count_per_spin(x), n_fermions_per_spin[np.newaxis, :]
                ).all(axis=-1)

        self._n_fermions_per_spin = n_fermions_per_spin
        self._n_fermions = n_fermions

        if spin_degrees == 1 or n_fermions_per_spin is None:
            super().__init__(n_orbitals, n_fermions=n_fermions)
        else:
            super().__init__(n_orbitals, extra_constraint_fn=constraint_fn)

    @property
    def spinless(self):
        return self._s == 0

    @property
    def n_fermions_per_spin(self):
        return self._n_fermions_per_spin

    @property
    def constrained_per_spin(self):
        return self.n_fermions_per_spin is not None

    @property
    def n_sites(self):
        return self._n_sites

    @property
    def n_fermions(self):
        ps = self.n_fermions_per_spin
        if ps is not None:
            return sum(ps)
        return self._n_fermions

    def count_per_spin(self, x):
        x_per_spin = np.split(x, self._spin_sections, axis=-1)
        n_per_spin = np.stack([np.sum(xs, axis=-1) for xs in x_per_spin], axis=-1)
        return n_per_spin

    def __repr__(self):
        return "LatticeFermions2nd(n_sites={}, s={})".format(self._n_sites, self._s)
