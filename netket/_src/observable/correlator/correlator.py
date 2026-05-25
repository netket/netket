# Copyright 2024 The NetKet Authors - All rights reserved.
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

"""Correlation function observables for lattice quantum systems."""

from __future__ import annotations

from netket.operator import AbstractOperator
from netket.operator._abstract_observable import AbstractObservable


class ConnectedCorrelator(AbstractObservable):
    r"""Connected two-point correlator :math:`\langle A B \rangle - \langle A \rangle \langle B \rangle`.

    Computes the connected correlator for a pair of operators (A, B) using
    three local-estimator channels :math:`[L_A, L_B, L_{AB}]` and the delta
    method to propagate statistical errors correctly.

    The correct estimator for :math:`\langle AB \rangle` is always
    :math:`L_{AB}(\sigma) = \sum_{\sigma'} \langle \sigma | AB | \sigma' \rangle
    \psi(\sigma') / \psi(\sigma)`, computed via the product operator ``A @ B``.

    Example::

        hi = nk.hilbert.Spin(s=1/2, N=4)
        A = nk.operator.spin.sigmax(hi, 0)
        B = nk.operator.spin.sigmax(hi, 1)
        obs = nk.observable.ConnectedCorrelator(A, B)
        result = vs.expect(obs)  # Stats: mean = <AB> - <A><B>
    """

    def __init__(self, op_A: AbstractOperator, op_B: AbstractOperator):
        r"""
        Args:
            op_A: First operator A.
            op_B: Second operator B. Must share the same Hilbert space as ``op_A``.
        """
        if op_A.hilbert != op_B.hilbert:
            raise ValueError(
                f"Both operators must share the same Hilbert space, "
                f"got {op_A.hilbert} and {op_B.hilbert}."
            )
        super().__init__(op_A.hilbert)
        self._op_A = op_A
        self._op_B = op_B
        self._product_op = op_A @ op_B

    @property
    def op_A(self) -> AbstractOperator:
        return self._op_A

    @property
    def op_B(self) -> AbstractOperator:
        return self._op_B

    @property
    def product_op(self) -> AbstractOperator:
        return self._product_op

    def __repr__(self):
        return f"ConnectedCorrelator(A={self._op_A}, B={self._op_B})"
