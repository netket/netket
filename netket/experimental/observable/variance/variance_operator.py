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


class VarianceObservable(AbstractObservable):
    r"""
    Observable computing the variance of a quantum operator :math:`O`.
    """

    def __init__(self, operator: AbstractOperator, use_Oloc_squared: bool = False):
        r"""
        Constructs the observable computing the variance of an arbitrary quantum operator :math:`O` as:

        .. math::

            \text{Var} = \frac{\langle \Psi | O^2 | \Psi \rangle}{\langle \Psi | \Psi \rangle} - \bigg( \frac{\langle \Psi | O | \Psi \rangle}{\langle \Psi | \Psi \rangle}\bigg)^2

        It can compute the first term using either the estimator of the squared operator :math:`O^2` (more precise but less efficient, since it requires
        the connected configurations and the matrix elements of :math:`O^2`):

        .. math::

            \text{Var} = \mathbb{E}_{\sigma \sim |\Psi(\sigma)|^2}\bigg[\frac{\langle \sigma | O^2 | \Psi \rangle}{\langle \sigma | \Psi \rangle}\bigg] - \bigg(\mathbb{E}_{\sigma \sim |\Psi(\sigma)|^2}\bigg[\frac{\langle \sigma | O | \Psi \rangle}{\langle \sigma | \Psi \rangle}\bigg]\bigg)^2.

        or using the square modulus of the estimator of :math:`O` (more noisy but more efficient):

        .. math::

            \text{Var} = \mathbb{E}_{\sigma \sim |\Psi(\sigma)|^2}\bigg[\bigg(\frac{\langle \sigma | O | \Psi \rangle}{\langle \sigma | \Psi \rangle} - \mathbb{E}_{\sigma \sim |\Psi(\sigma)|^2}\bigg[\frac{\langle \sigma | O | \Psi \rangle}{\langle \sigma | \Psi \rangle}\bigg]\bigg)^2\bigg]

        This VariationalOperator wraps an operator such that the gradient will not be computed
        with respect to the expectation value, but with respect to the variance.

        Args:
            operator: The operator for which the variance is to be computed.
            use_Oloc_squared: (Defaults False) if True, uses the fast estimator obtained by squaring the local estimator O_loc. If False, uses the straightforward estimator of (O@O)_loc which is quadratically more expensive. The fast estimator can sometimes lead to worse quality results.

        Returns:
            Observable computing the variance of `operator`.
        """
        super().__init__(operator.hilbert)
        self._operator = operator

        if use_Oloc_squared:
            self._operator_squared = nk.operator.Squared(operator)
        else:
            self._operator_squared = operator @ operator

    @property
    def operator(self) -> AbstractOperator:
        """
        The operator for which the variance is to be computed.
        """
        return self._operator

    @property
    def operator_squared(self) -> AbstractOperator:
        """
        The squared of the operator for which the variance is to be computed.
        Depending on the flag `use_Oloc_squared`, this can be the operator using the local
        estimator of `O^2` (False), or the one using the square modulus of the
        local estimator of `O` (True).
        """
        return self._operator_squared

    def __repr__(self):
        return f"VarianceObservable(op={self.operator})"
