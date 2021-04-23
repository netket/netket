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

from typing import Callable, Optional, Union, Tuple, Any
from functools import partial

import jax
import flax
from jax import numpy as jnp
from flax import struct

from plum import dispatch

from netket.utils.types import PyTree, Array


@struct.dataclass
class SR:
    """
    Base class holding the parameters for the way to find the solution to
    (S + I*diag_shift) / F.
    """

    diag_shift: float = 0.01
    """Diagonal shift added to the S matrix."""


@struct.dataclass
class AbstractSMatrix:
    """
    S matrix base class.
    This can either be a jnp matrix, a lazy wrapper, or anything, as long as
    it satisfies this basic API.
    """

    sr: SR
    """Parameters for the solution of the system."""

    def __matmul__(self, vec):
        raise NotImplementedError()

    def __rtruediv__(self, y):
        return self.solve(y)

    def solve(self, y: PyTree, x0: Optional[PyTree] = None, **kwargs) -> PyTree:
        """
        Solve the linear system x=⟨S⟩⁻¹⟨y⟩ with the chosen iterataive solver.

        Args:
            y: the vector y in the system above.
            x0: optional initial guess for the solution.

        Returns:
            x: the PyTree solving the system.
            info: optional additional informations provided by the solver. Might be
                None if there are no additional informations provided.
        """
        return self.__rtruediv__(y)

    @jax.jit
    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.s

        Returns:
            A dense matrix representation of this S matrix.
        """
        raise NotImplementedError()


@dispatch.annotations()
def SMatrix(sr: SR, vstate: object) -> AbstractSMatrix:
    """
    Construct the S matrix given a
    """
    raise NotImplementedError
