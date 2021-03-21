import numbers
from typing import List

import numpy as np

from ._abstract_operator import AbstractOperator
from ._local_operator import LocalOperator
from netket.hilbert import DoubledHilbert, AbstractHilbert


class AbstractSuperOperator(AbstractOperator):
    """
    Generic base class for super-operators acting on the tensor product (DoubledHilbert)
    space ℋ⊗ℋ, where ℋ is the physical space.

    Behaves on :ref:`netket.variational.VariationalMixedState` as normal operators behave
    on :ref:`netket.variational.VariationalState`.
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
