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
import abc
from typing import Optional, Callable

from netket.utils.types import DType, PyTree, Array

from netket.hilbert import AbstractHilbert
from netket.operator import AbstractOperator


class ContinuousOperator(AbstractOperator):
    r"""This class is the abstract base class for operators defined on a
    continuous Hilbert space. Users interested in implementing new
    quantum Operators for continuous Hilbert spaces should subclass
    `ContinuousOperator` and implement its interface.
    """

    def __init__(self, hilbert: AbstractHilbert, dtype: DType = float):
        r"""
        Constructs the continuous operator acting on the given hilbert space and
        with a certain data type.

        Args:
            hilbert: The underlying Hilbert space on which the operator is defined
            dtype: Data type of the matrix elements. Defaults to `np.float64`
        """

        self._dtype = dtype
        super().__init__(hilbert)

    @property
    def dtype(self) -> DType:
        return self._dtype

    @abc.abstractmethod
    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):
        r"""This method defines the action of the local operator on a given quantum state
        `logpsi` for a given congfiguration `x`.
        :math:`O_{loc}(x) =  \frac{\bra{x}O{\ket{\psi}}{\bra{x}\ket{\psi}}`
        This method is executed inside of a `jax.jit` block.
        Any static data from the operator itself should be captured in the method.
        Any additional data is provided by the `_pack_arguments`-method
        and will be passed as the `data` argument in this method (Example: masses in kinetic energy).

        Args:
            logpsi: variational state
            params: parameters for the variational state
            x: a sample of particle positions
            data: additional data
        """

    @abc.abstractmethod
    def _pack_arguments(self) -> Optional[PyTree]:
        r"""This methods should return a PyTree that will be passed as the `data` argument
        to the `_expect_kernel`. The PyTree should be composed of jax arrays or hashable
        objects.

        For example for the kinetic energy this method would return the masses of the
        individual particles."""

    def __add__(self, other):
        if isinstance(self, ContinuousOperator) and isinstance(
            other, ContinuousOperator
        ):
            from netket.operator import SumOperator

            return SumOperator(self, other)
        else:
            return NotImplemented  # pragma: no cover

    def __rmul__(self, other):
        if isinstance(self, ContinuousOperator) and isinstance(other, float):
            return self * other
        else:
            return NotImplemented  # pragma: no cover

    def __mul__(self, other):
        if isinstance(self, ContinuousOperator) and isinstance(other, float):
            from netket.operator import SumOperator

            return SumOperator(self, coefficients=other)
        else:
            return NotImplemented  # pragma: no cover
