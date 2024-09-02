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


class SymmExpSum(nn.Module):
    r"""
    A flax module symmetrizing the log-wavefunction :math:`\log\psi_\theta(\sigma)`
    encoded into another flax module (:class:`flax.linen.Module`) by summing over
    all possible symmetries :math:`g` in a certain discrete permutation
    group :math:`G`.

    .. math::

        \log\psi_\theta(\sigma) = \frac{1}{|G|}\log\sum_{g\in G}
            \chi_g\exp[\log\psi_\theta(T_{g}\sigma)]

    For the ground-state, it is usually found that :math:`\chi_g=1 \forall g\in G`.

    To construct this network, one has to specify the module, the symmetry group
    and (optionally)the id of the character to consider.

    The module's :code:`.__call__` will be called.
    The :code:`symm_group` attribute

    Examples:

       Constructs a :ref:`netket.nn.blocks.SymmExpSum` for a bare
       :ref:`netket.models.RBM`, summing over all translations of a
       2D Square lattice

       >>> import netket as nk
       >>> graph = nk.graph.Square(3)
       >>> print("number of translational symmetries: ", len(graph.translation_group()))
       number of translational symmetries:  9
       >>> # Construct the bare unsymmetrized machine
       >>> machine_no_symm = nk.models.RBM(alpha=2)
       >>> # Symmetrize the RBM over all translations
       >>> ma = nk.nn.blocks.SymmExpSum(module = machine_no_symm, symm_group=graph.translation_group())

       If you have a Convolutional NN that is already invariant under translations, you might
       want to only symmetrize over the point-group (mirror symmetry and rotations).

       >>> import netket as nk
       >>> graph = nk.graph.Square(3)
       >>> print("number of point-group symmetries: ", len(graph.point_group()))
       number of point-group symmetries:  8
       >>> # Construct the bare unsymmetrized machine
       >>> machine_no_symm = nk.models.RBM(alpha=2)
       >>> # Symmetrize the RBM over all translations
       >>> ma = nk.nn.blocks.SymmExpSum(module = machine_no_symm, symm_group=graph.point_group())

    """

    module: nn.Module
    """The neural network architecture encoding the log-wavefunction
    to symmetrize in the :code:`.__call__` function."""

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

    character_id: int | None = None
    """The # identifying the target character in the character table of
    the symmetry group. By default the characters are taken to be all
    `1`, giving the homogeneous state.

    The characters are accessed as:

    .. code::

        symm_group.character_table()[character_id]

    """

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

        # Extract the characters. Those are compile-time constant (a numpy array).
        characters: np.ndarray
        if self.character_id is None:
            characters = np.ones(len(np.asarray(self.symm_group)))
        else:
            characters = self.symm_group.character_table()[self.character_id]

        characters = characters.reshape((-1,) + tuple(1 for _ in range(x.ndim - 1)))

        # If those are all positive, then use standard logsumexp that returns a
        # real-valued, positive logsumexp
        logsumexp_fun = (
            jax.scipy.special.logsumexp if np.all(characters >= 0) else logsumexp_cplx
        )

        # log (sum_i ( c_i/Nsymm* exp(psi[sigma_i])))
        psi = logsumexp_fun(psi_symm, axis=0, b=characters / len(self.symm_group))
        return psi
