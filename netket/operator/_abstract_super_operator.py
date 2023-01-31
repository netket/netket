# Copyright 2021 The NetKet Authors - All rights reserved.
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

import numpy as np
import jax.numpy as jnp

from netket.hilbert import DoubledHilbert, AbstractHilbert

from ._discrete_operator import DiscreteOperator


class AbstractSuperOperator(DiscreteOperator):
    """
    Generic base class for super-operators acting on the tensor product (DoubledHilbert)
    space ℋ⊗ℋ, where ℋ is the physical space.

    Behaves on :ref:`netket.vqs.VariationalMixedState` as normal operators behave
    on :ref:`netket.vqs.VariationalState`.
    Cannot be used to act upon pure states.
    """

    def __init__(self, hilbert):
        """
        Initialize a super-operator by passing it the physical hilbert space on which it acts.

        This init method constructs the doubled-hilbert space and pass it down to the fundamental
        abstractoperator.
        """
        super().__init__(DoubledHilbert(hilbert))

    @property
    def hilbert_physical(self) -> AbstractHilbert:
        """The physical hilbert space on which this super-operator acts."""
        return self.hilbert.physical

    def to_qobj(self):  # -> "qutip.Qobj"
        raise NotImplementedError("Superoperator to Qobj not yet implemented")

    def __matmul__(self, other):
        # Override DiscreteOperator to implement the Squared trick.
        # Should eventually remove it as well.
        if isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray):
            return self.apply(other)
        elif isinstance(other, AbstractSuperOperator):
            if self == other and self.is_hermitian:
                from ._lazy import Squared

                return Squared(self)
            else:
                return self._op__matmul__(other)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        # override DiscreteOperator to implement the Squared trick.
        # Should eventually remove it as well.
        if isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray):
            return NotImplemented
        elif isinstance(other, AbstractSuperOperator):
            if self == other and self.is_hermitian:
                from ._lazy import Squared

                return Squared(self)
            else:
                return self._op__rmatmul__(other)
        else:
            return NotImplemented
