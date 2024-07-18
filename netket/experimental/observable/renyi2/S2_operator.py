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

from textwrap import dedent

import jax.numpy as jnp
import numpy as np

from netket.hilbert import HomogeneousHilbert

from netket.operator._abstract_observable import AbstractObservable


class Renyi2EntanglementEntropy(AbstractObservable):
    r"""
    Rényi2 bipartite entanglement entropy of a state :math:`| \Psi \rangle`
    between partitions A and B.

    """

    def __init__(self, hilbert: None, partition: jnp.array):
        r"""
        Constructs the operator computing the Rényi2 entanglement entropy of
        a state :math:`| \Psi \rangle` for a partition with partition A:

        .. math::

            S_2 = -\log_2 \text{Tr}_A [\rho^2]

        where :math:`\rho = | \Psi \rangle \langle \Psi |` is the density
        matrix of the system and :math:`\text{Tr}_A` indicates the partial
        trace over the partition A.

        The Monte Carlo estimator of S_2 [Hastings et al., PRL 104, 157201 (2010)] is:

        .. math::

            S_2 = - \log \langle \frac{\Psi(\sigma,\eta^{\prime}) \Psi(\sigma^{\prime},\eta)}{\Psi(\sigma,\eta) \Psi(\sigma^{\prime},\eta^{\prime})} \rangle

        where the mean is taken over the distribution
        :math:`\Pi(σ,η) \Pi(σ',η')`, :math:`\sigma \in A`,
        :math:`\eta \in \bar{A}` and
        :math:`\Pi(\sigma, \eta) = |\Psi(\sigma,\eta)|^2 / \langle \Psi | \Psi \rangle`.

        Args:
            hilbert: hilbert space of the system.
            partition: list of the indices identifying the degrees of
                freedom in one partition of the full system. All
                indices should be integers between 0 and hilbert.size

        Returns:
            Rényi2 operator for which computing the expected value.
        """

        # Homogeneos, not discrete... because we don't support
        # tensorhilbert and such. We could generalize easily by
        # setting psi(x not in hilbert) = 0
        if not isinstance(hilbert, HomogeneousHilbert):
            raise TypeError(
                dedent(
                    """
                    Entanglement Entropy estimation is only implemented for
                    Homogeneous discrete Hilbert spaces.

                    It can be easily generalised to continuous spaces, so if
                    you want this feature get in touch with us!"
                    """
                )
            )
        else:
            if hilbert.constrained:
                raise ValueError(
                    dedent(
                        """
                        Entanglement entropy estimation is not implemented
                        for constrained Hilbert spaces.

                        It can be generalised, so get in touch with us if you
                        need this feature.
                        """
                    )
                )

        super().__init__(hilbert)

        self._partition = np.array(list(set(partition)))

        if (
            np.where(self._partition < 0)[0].size > 0
            or np.where(self._partition > hilbert.size - 1)[0].size > 0
        ):
            raise ValueError(
                "Invalid partition: possible negative indices or indices outside the system size."
            )

    @property
    def partition(self):
        r"""
        list of indices for the degrees of freedom in the partition
        """
        return self._partition

    def __repr__(self):
        return f"Renyi2EntanglementEntropy(hilbert={self.hilbert}, partition={self.partition})"
