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

import jax.numpy as jnp

from netket.utils import struct
from netket.utils.types import Scalar


@struct.dataclass
class KahanSum(struct.Pytree):
    """
    Accumulator implementing Kahan summation [1], which reduces
    the effect of accumulated floating-point error.

    [1] https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    """

    value: Scalar
    """
    Current value stored in this array
    """
    compensator: Scalar
    """
    Compensator used to fix addition/subtraction
    """

    def __init__(self, value, compensator=None) -> None:
        """
        Constructs the Kahan Summation accumulator with a zero-initialized
        compensator if unspecified.

        Args:
            value: The value to initialize this scalar.
            compensator: The state of the compensator. 0 by default.
        """
        self.value = jnp.asarray(value)
        if compensator is None:
            compensator = jnp.zeros_like(self.value)
        self.compensator = compensator

    def __add__(self, other: Scalar):
        delta = other - self.compensator
        new_value = self.value + delta
        new_compensator = (new_value - self.value) - delta
        return KahanSum(new_value, new_compensator)

    def __jax_array__(self, dtype=None):
        return self.value.astype(dtype)
