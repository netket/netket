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

from typing import Optional

import jax.numpy as jnp
import numpy as np

from netket.operator import AbstractOperator
from netket.hilbert import DiscreteHilbert
from netket.utils.types import DType


class Renyi2EntanglementEntropy(AbstractOperator):

    r"""
    Rényi2 entanglement entropy of a state :math:`| \Psi \rangle` for a partition with subsystem A.

    """

    def __init__(
        self,
        hilbert: None,
        subsystem: jnp.array,
        *,
        dtype: Optional[DType] = None,
    ):
        r"""
        Constructs the operator computing the Rényi2 entanglement entropy of a state :math:`| \Psi \rangle` for a partition with subsystem A:

        .. math::

            S_2 = -\log_2 \text{Tr}_A [\rho^2]

        where :math:`\rho = | \Psi \rangle \langle \Psi |` is the density matrix of the system and :math:`\text{Tr}_A` indicates the partial trace over the subsystem A.

        The Monte Carlo estimator of S_2 [Hastings et al., PRL 104, 157201 (2010)] is:

        .. math::

            S_2 = - \log \langle \frac{\Psi(\sigma,\eta^{\prime}) \Psi(\sigma^{\prime},\eta)}{\Psi(\sigma,\eta) \Psi(\sigma^{\prime},\eta^{\prime})} \rangle

        where the mean is taken over the distribution :math:`\Pi(σ,η) \Pi(σ',η')`, :math:`\sigma \in A`, :math:`\eta \in \bar{A}` and :math:`\Pi(\sigma, \eta) = |\Psi(\sigma,\eta)|^2 / \langle \Psi | \Psi \rangle`.

        Args:
            hilbert: hilbert space of the system.
            subsystem: list of the indices identifying the degrees of freedom in one subsystem of the full system.
                All indices should be integers between 0 and hilbert.size

        Returns:
            Rényi2 operator for which computing the expected value.
        """

        if not isinstance(hilbert, DiscreteHilbert):
            raise TypeError(
                "Entanglement Entropy estimation is only implemented for Discrete Hilbert spaces. It can be easily generalised to continuous spaces, so if you want this feature get in touch with us!"
            )

        super().__init__(hilbert)

        self._dtype = dtype
        self._subsystem = np.sort(np.array(subsystem))

        if (
            self._subsystem.size > hilbert.size
            or np.where(self._subsystem < 0)[0].size > 0
            or np.where(self._subsystem > hilbert.size)[0].size > 0
        ):
            print("Invalid partition")

    @property
    def dtype(self):
        return self._dtype

    @property
    def subsystem(self):
        r"""
        list of indices for the degrees of freedom in the subsystem
        """
        return self._subsystem

    @property
    def is_hermitian(self):
        return True

    def __repr__(self):
        return f"Renyi2EntanglementEntropy(hilbert={self.hilbert}, subsys={self.subsystem})"
