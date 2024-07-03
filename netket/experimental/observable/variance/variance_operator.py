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

import netket as nk
from netket.operator import AbstractOperator

from netket.operator._abstract_observable import AbstractObservable


class VarianceOperator(AbstractObservable):
    r"""
    Observable corresponding to the variance of a an arbitrary quantum operator O:

    .. math::

        \text{Var} = \frac{\langle \Psi | O^2 | \Psi \rangle}{\langle \Psi | \Psi \rangle} - \bigg( \frac{\langle \Psi | O | \Psi \rangle}{\langle \Psi | \Psi \rangle}\bigg)^2

    It can compute the variance using either the estimator of the squared operator :math:`O^2` (more precise but less efficient, since it requires
    the connected configurations and the matrix elements of :math:`O^2`):

    .. math::

        \text{Var} = \mathbb{E}_{\sigma \sim |\Psi(\sigma)|^2}\bigg[\frac{\langle \sigma | O^2 | \Psi \rangle}{\langle \sigma | \Psi \rangle}\bigg] - \bigg(\mathbb{E}_{\sigma \sim |\Psi(\sigma)|^2}\bigg[\frac{\langle \sigma | O | \Psi \rangle}{\langle \sigma | \Psi \rangle}\bigg]\bigg)^2

    or using the square modulus of the estimator of O (more noisy but more efficient, since it involves only :math:`O`):

    .. math::

        \text{Var} = \mathbb{E}_{\sigma \sim |\Psi(\sigma)|^2}\bigg[\bigg(\frac{\langle \sigma | O | \Psi \rangle}{\langle \sigma | \Psi \rangle} - \mathbb{E}_{\sigma \sim |\Psi(\sigma)|^2}\bigg[\frac{\langle \sigma | O | \Psi \rangle}{\langle \sigma | \Psi \rangle}\bigg]\bigg)^2\bigg]

    """

    def __init__(self, op: AbstractOperator, use_Oloc2: bool=False):
        """
        Constructs the VariationalOperator wrapping an operator such that the gradient will not be computed
        with respect to the expectation value, but with respect to the variance.

        Args:
            op: The operator for which the variance is to be computed.
            use_Oloc2: If True, uses the local estimator of the squared operator `O^2` for variance computation.
                        If False, uses only the operator `O` for variance computation (defaults to False).

        """
        super().__init__(op.hilbert)
        self._op = op

        if use_Oloc2:
            self._op2 = nk.operator.Squared(op)
        else:
            self._op2 = op @ op

    @property
    def op(self) -> AbstractOperator:
        """
        The opeator for which the variance is to be computed.
        """
        return self._op

    @property
    def op2(self)-> AbstractOperator:
        """
        The squared of the operator for which the variance is to be computed.
        Depending on the flag `use_Oloc2`, this can be the operator using the local 
        estimator of `O^2` (True), or the one using the square modulus of the
        local estimator of `O` (False).
        """
        return self._op2

    def __repr__(self):
        return f"VarianceOperator(op={self.op})"
