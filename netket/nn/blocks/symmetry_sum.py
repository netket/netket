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


import jax
import numpy as np

from flax import linen as nn

from netket.jax import logsumexp_cplx
from netket.utils.group import PermutationGroup
from netket.utils import HashableArray


class SymmExpSum(nn.Module):
    r"""
    A wrapper module to symmetrise a variational wave function with respect
    to a permutation group :math:`G` by summing over all permuted inputs.

    Given the characters :math:`\chi_g` of an irreducible representation
    (irrep) of :math:`G`, a variational state that transforms according
    to this irrep is given by the projection formula

    .. math::

        \psi_\theta(\sigma) = \frac{1}{|G|}\sum_{g\in G}
            \chi_g \psi_\theta(T_{g}\sigma).

    Ground states usually transform according to the trivial irrep
    :math:`\chi_g=1 \forall g\in G`.

    Examples:

        Symmetrise an :ref:`netket.models.RBM` with respect to the space
        group of a 2D :ref:`netket.graph.Square` lattice:

        >>> import netket as nk
        >>> graph = nk.graph.Square(4)
        >>> group = graph.space_group()
        >>> print("Size of space group:", len(group))
        Size of space group: 128
        >>> # Construct the bare unsymmetrized machine
        >>> machine_no_symm = nk.models.RBM(alpha=2)
        >>> # Symmetrize the RBM over the space group
        >>> ma = nk.nn.blocks.SymmExpSum(module = machine_no_symm, symm_group=group)

        Nontrivial irreps can be specified using momentum and point-group
        quantum numbers:

        >>> from math import pi
        >>> print(group.little_group(pi, 0).character_table_readable())
        (['1xId()', '1xRefl(0°)', '1xRefl(90°)', '1xRot(180°)'],
        array([[ 1.,  1.,  1.,  1.],
               [ 1.,  1., -1., -1.],
               [ 1., -1.,  1., -1.],
               [ 1., -1., -1.,  1.]]))
        >>> chi = group.space_group_irreps(pi, 0)[1]
        >>> ma = nk.nn.blocks.SymmExpSum(module = machine_no_symm, symm_group=group, characters=chi)

        Convolutional networks are already invariant under translations,
        so they only need to be symmetrised with respect to the point group
        (e.g., mirrors and rotations).

        >>> import netket as nk
        >>> graph = nk.graph.Square(4)
        >>> print("Size of the point group:", len(graph.point_group()))
        Size of the point group: 8
        >>> # Construct a translation-invariant RBM
        >>> machine_trans = nk.models.RBMSymm(alpha=2, symmetries=graph.translation_group())
        >>> # Symmetrize the RBM over the point group
        >>> ma = nk.nn.blocks.SymmExpSum(module = machine_trans, symm_group=graph.point_group())
    """

    module: nn.Module
    """The unsymmetrised neural-network ansatz."""

    symm_group: PermutationGroup
    """The symmetry group to use. It should be a valid
    :ref:`netket.utils.group.PermutationGroup` object.

    Can be extracted from a :ref:`netket.graph.Lattice` object by calling
    :meth:`~netket.graph.Lattice.point_group` or
    :meth:`~netket.graph.Lattice.translation_group`.

    Alternatively, if you have a :class:`netket.graph.Graph` object you
    can build it from :meth:`~netket.graph.Lattice.automorphisms`.

    .. code::

        graph = nk.graph.Square(3)
        symm_group = graph.point_group()

    """

    characters: HashableArray | None = None
    r"""Characters :math:`\chi_g` of the space group to project onto.

    Only one of `characters` and `character_id` may be specified.
    If neither is specified, the character is taken to be all 1,
    yielding a trivially symmetric state.
    """

    character_id: int | None = None
    """Index of the character to project onto in the character table
    of the symmetry group.

    The characters are accessed as:

    .. code::

        symm_group.character_table()[character_id]

    Only one of `characters` and `character_id` may be specified.
    If neither is specified, the character is taken to be all 1,
    yielding a trivially symmetric state.
    """

    def setup(self):
        if self.characters is None:
            if self.character_id is None:
                self._chi = np.ones(len(np.asarray(self.symm_group)))
            else:
                self._chi = self.symm_group.character_table()[self.character_id]
        else:
            if self.character_id is None:
                self._chi = self.characters.wrapped
            else:
                raise AttributeError(
                    "Must not specify both `characters` and `character_id`"
                )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Accepts a single input or arbitrary batch of inputs.

        The last dimension of x must match the shape of the permutation
        group.
        """
        # apply the group and obtain a x_symm of shape (N_symm, ...)
        x_symm = self.symm_group @ x
        # reshape it to (-1, N_sites)
        x_symm_shape = x_symm.shape
        x_symm = x_symm.reshape(-1, x.shape[-1])

        # Compute the log-wavefunction obtaining (-1,) and reshape to (N_symm, ...)
        psi_symm = self.module(x_symm).reshape(*x_symm_shape[:-1])

        characters = np.expand_dims(self._chi, tuple(range(1, x.ndim)))

        # If those are all positive, then use standard logsumexp that returns a
        # real-valued, positive logsumexp
        logsumexp_fun = (
            jax.scipy.special.logsumexp if np.all(characters >= 0) else logsumexp_cplx
        )

        # log (sum_i ( c_i/Nsymm* exp(psi[sigma_i])))
        psi = logsumexp_fun(psi_symm, axis=0, b=characters / len(self.symm_group))
        return psi
