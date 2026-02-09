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


from abc import ABC


from netket.hilbert import TensorHilbert

from .._abstract_operator import AbstractOperator
from .._discrete_operator_jax import DiscreteJaxOperator


class EmbedOperator(ABC):
    def __new__(cls, hi, op, *args, **kwargs):
        # This logic overrides the constructor, such that if someone tries to
        # construct this class directly by calling `SumOperator(...)`
        # it will construct either a DiscreteHilbert or TensorDiscreteHilbert
        from .operator import EmbedGenericOperator

        # from .discrete_operator import SumDiscreteOperator
        from .discrete_jax_operator import EmbedDiscreteJaxOperator

        # from .continuous import SumContinuousOperator

        if cls is EmbedOperator:
            if isinstance(op, DiscreteJaxOperator):
                cls = EmbedDiscreteJaxOperator
            # elif isinstance(op, DiscreteOperator):
            #     cls = SumDiscreteOperator
            # elif isinstance(op, ContinuousOperator):
            #     cls = SumContinuousOperator
            else:
                cls = EmbedGenericOperator
        return super().__new__(cls)

    def __init__(
        self,
        hilbert: TensorHilbert,
        operator: AbstractOperator,
        subspace: int,
        **kwargs,
    ):
        r"""Embeds an operator acting on a subspace into a larger tensor product Hilbert space.

        Mathematically, this operator represents:

        .. math::

            \hat{O}_{\text{embed}} = \mathbb{I}_0 \otimes \mathbb{I}_1 \otimes \cdots
                \otimes \hat{O}_i \otimes \cdots \otimes \mathbb{I}_N

        where :math:`\hat{O}_i` is the operator acting on subspace :math:`i`, and
        :math:`\mathbb{I}_j` are identity operators on all other subspaces :math:`j \neq i`.

        This is useful for constructing operators on composite quantum systems, such as
        coupled electron-phonon systems, where you want an operator to act only on
        one subsystem (e.g., only on electrons or only on phonons) while leaving
        the other subsystems unchanged.

        Args:
            hilbert: A TensorHilbert space representing the composite system.
                Must be a tensor product of multiple Hilbert spaces.
            operator: The operator to embed. Its Hilbert space must match
                the specified subspace of the tensor Hilbert space.
            subspace: The index of the subspace (in the TensorHilbert) where
                the operator acts. Must satisfy
                ``hilbert.subspaces[subspace] == operator.hilbert``.

        Warning:
            **Performance consideration for Hamiltonian construction:**

            When building complex Hamiltonians, the order of operations matters significantly
            for performance. Operators should be simplified *before* embedding to avoid
            creating many redundant embedded operators that cannot be combined, leading to
            very high connected elements count.

            Try to always simplify first, then embed:

            >>> # Build the full operator on the subspace first
            >>> h_fermion = sum(nk.operator.boson.number(hi_boson, i) for i in range(N))
            >>> h_embed = nk.operator.EmbedOperator(hi_joint, h_fermion, subspace=0)
            >>> H = H + h_embed  # Single embedded operator

            as opposed to this **negative example below**

            >>> # Don't do this! Creates many embedded operators that can't be simplified
            >>> for i in range(N):
            >>>     h_fermion = nk.operator.boson.number(hi_boson, i)
            >>>     # Embedding inside the loop prevents simplification!
            >>>     h_embed = nk.operator.EmbedOperator(hi_joint, h_fermion, subspace=0)
            >>>     H = H + h_embed  # Many redundant embedded operators!

            The first approach creates a single embedded operator with optimized connected
            elements, while the second creates N separate embedded operators that cannot
            be combined, resulting in significantly more connected states and slower
            evaluation.

        Examples:
            Embed a spin operator in a composite spin-boson system:

            >>> import netket as nk
            >>> # Create subsystems
            >>> hi_spin = nk.hilbert.Spin(s=1/2, N=2)
            >>> hi_boson = nk.hilbert.Fock(n_max=3, N=2)
            >>> hi_joint = nk.hilbert.TensorHilbert(hi_spin, hi_boson)
            >>> # Create operator on spin subsystem
            >>> sx = nk.operator.spin.sigmax(hi_spin, 0)
            >>> # Embed it in the joint space (acts on subspace 0)
            >>> sx_embed = nk.operator.EmbedOperator(hi_joint, sx, subspace=0)

            Combine embedded operators from different subspaces:

            >>> # Bosonic operator on second subspace
            >>> n_b = nk.operator.boson.number(hi_boson, 0)
            >>> n_b_embed = nk.operator.EmbedOperator(hi_joint, n_b, subspace=1)
            >>> # Combine them
            >>> H = sx_embed + 0.5 * n_b_embed  # Spin + phonon Hamiltonian
            >>> H_interaction = sx_embed @ n_b_embed  # Spin-phonon coupling
        """
        if not isinstance(hilbert, TensorHilbert):
            raise TypeError(
                "The hilbert space of an EmbedOperator must be a TensorHilbert."
            )
        if not hilbert.subspaces[subspace] == operator.hilbert:
            raise TypeError(
                f"The {subspace}-th hilbert space of the tensor hilbert {hilbert} does not match"
                f" the hilbert space of the operator {operator}."
            )

        # operators, coefficients = _flatten_sumoperators(operators, coefficients)

        self._operator = operator
        self._subspace = subspace
        self._dtype = operator.dtype

        super().__init__(
            hilbert,
        )  # forwards all unused arguments so that this class is a mixin.

    @property
    def dtype(self):
        return self._dtype

    @property
    def operator(self) -> AbstractOperator:
        """The tuple of all operators in the terms of this sum. Every
        operator is summed with a corresponding coefficient
        """
        return self._operator

    @property
    def subspace(self) -> int:
        """The index of the subspace in the hilbert space."""
        return self._subspace

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.hilbert} on {self.subspace} : {self.operator})"

    def __add__(self, other):
        if not isinstance(other, EmbedOperator):
            return super().__add__(other)

        if self.hilbert != other.hilbert:
            raise ValueError("Cannot add EmbedOperators with different hilbert spaces.")
        elif (
            self.operator.hilbert == other.operator.hilbert
            and self.subspace == other.subspace
        ):
            # if same hilbert but different subspaces
            return EmbedOperator(
                self.hilbert,
                self.operator + other.operator,
                self.subspace,
            )
        else:
            from netket.operator._sum import SumOperator

            return SumOperator(self, other)

    def _op__matmul__(self, other):
        if self.hilbert != other.hilbert:
            raise ValueError(
                f"Cannot multiply operators on different Hilbert spaces: "
                f"{self.hilbert} vs {other.hilbert}"
            )

        if isinstance(other, EmbedOperator) and self.subspace == other.subspace:
            return EmbedOperator(
                self.hilbert, self.operator @ other.operator, subspace=self.subspace
            )
        else:
            from netket.operator._prod import ProductOperator

            return ProductOperator(other, self)
