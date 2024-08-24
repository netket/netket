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
import jax.numpy as jnp

from netket.utils import struct
from netket.utils.types import Scalar, Array

from .base import (
    DiscreteHilbertConstraint,
)


class SumOnPartitionConstraint(DiscreteHilbertConstraint):
    """
    Constraint of an Hilbert space enforcing a total sum of all the values
    in partitions of different sizes.

    Constructed by specifying the tuples of sums on every partition and the length
    of every partition.

    This can be used to represent tensor products of Spin subsystems with
    different total magnetizations, or constraint on the number of fermions
    on different parts.
    """

    sum_values: tuple[Scalar, ...] = struct.field(pytree_node=False)
    sizes: tuple[int, ...] = struct.field(pytree_node=False, default=None)

    def __init__(self, sum_values, sizes):
        if not (isinstance(sum_values, tuple) and isinstance(sizes, tuple)):
            raise TypeError("sum_values and sizes must be tuples.")
        if not len(sum_values) == len(sizes):
            raise ValueError("Length mismatch between sum values and sizes")
        if any(v is None for v in sum_values):
            raise TypeError("None not supported as a sum constraint.")

        self.sum_values = sum_values
        self.sizes = sizes

    @jax.jit
    def __call__(self, x: Array) -> Array:
        s0 = 0
        conditions = []
        for N, sv in zip(self.sizes, self.sum_values):
            conditions.append(x[..., s0 : s0 + N].sum(axis=-1) == sv)
            s0 = s0 + N
        return jax.tree_util.tree_reduce(jnp.logical_and, conditions)

    def __hash__(self):
        return hash(("SumOnPartitionConstraint", self.sum_values, self.sizes))

    def __eq__(self, other):
        if isinstance(other, SumOnPartitionConstraint):
            return self.sum_values == other.sum_values and self.sizes == other.sizes
        return False

    def __repr__(self):
        return f"SumOnPartitionConstraint(sum_values={self.sum_values}, sizes={self.sizes})"
