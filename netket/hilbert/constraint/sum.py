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

import jax

from netket.utils import struct
from netket.utils.types import Scalar, Array

from .base import DiscreteHilbertConstraint


class SumConstraint(DiscreteHilbertConstraint):
    """
    Constraint of an Hilbert space enforcing a total sum of all the values in the degrees of freedom.

    Constructed by specifying the total sum. For Fock-like spaces this is the total population,
    while for Spin-like spaces this is the magnetisation.
    """

    sum_value: Scalar = struct.field(pytree_node=False)

    def __init__(self, sum_value: Scalar):
        if sum_value is None:
            raise TypeError("sum_value must be a number.")

        self.sum_value = sum_value

    @jax.jit
    def __call__(self, x: Array) -> Array:
        return x.sum(axis=-1) == self.sum_value

    def __hash__(self):
        return hash(("SumConstraint", self.sum_value))

    def __eq__(self, other):
        if isinstance(other, SumConstraint):
            return self.sum_value == other.sum_value
        return False

    def __repr__(self):
        return f"SumConstraint({self.sum_value})"
