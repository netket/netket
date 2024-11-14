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

from textwrap import dedent

import jax
import jax.numpy as jnp

from netket.utils import struct, dispatch

from .base import (
    DiscreteHilbertConstraint,
)


@dispatch.parametric
@struct.dataclass
class ExtraConstraint(DiscreteHilbertConstraint):
    """
    Wraps a Constraint and its HilbertIndex into a second constraint (with a
    default lookup-based hilbert index).

    The first argument is the base constraint while the second is the wrapping
    one.
    """

    base_constraint: DiscreteHilbertConstraint = struct.field(pytree_node=False)
    extra_constraint: DiscreteHilbertConstraint = struct.field(pytree_node=False)

    def __call__(self, x: jax.Array) -> jax.Array:
        conditions = jnp.stack(
            [self.base_constraint(x), self.extra_constraint(x)], axis=0
        )
        return jnp.all(conditions, axis=0)

    def __hash__(self):
        return hash(("ExtraConstraint", self.base_constraint, self.extra_constraint))

    def __eq__(self, other):
        if isinstance(other, ExtraConstraint):
            return (
                self.base_constraint == other.base_constraint
                and self.extra_constraint == other.extra_constraint
            )
        return False

    def __repr__(self):
        return f"ExtraConstraint({self.base_constraint}, {self.extra_constraint})"

    # ----- Parametric class definition
    # Definitions to make the @dispatch.parametric class give informative errors
    # Look at https://beartype.github.io/plum/parametric.html for more information

    @classmethod
    def __init_type_parameter__(self, baseT: type, extraT: type):
        """Check whether the type parameters are valid."""
        # In this case, we use `@dispatch` to check the validity of the type parameter.
        if not (
            issubclass(baseT, DiscreteHilbertConstraint)
            and issubclass(extraT, DiscreteHilbertConstraint)
        ):
            raise TypeError(
                dedent(
                    """The arguments of ExtraConstraint must both be subtypes of
                    nk.hilbert.constraint.DiscreteHilbertConstraint, but one or more are not.

                    To verify if those are valid type parameters you can check it with
                    issubclass(arg, nk.hilbert.constraint.DiscreteHilbertConstraint).
                    """
                )
            )
        return baseT, extraT

    @classmethod
    def __infer_type_parameter__(
        self,
        base_constraint: DiscreteHilbertConstraint,
        extra_constraint: DiscreteHilbertConstraint,
    ):
        """Inter the type parameter from the arguments."""
        if not (
            isinstance(base_constraint, DiscreteHilbertConstraint)
            and isinstance(extra_constraint, DiscreteHilbertConstraint)
        ):
            raise TypeError(
                dedent(
                    """The arguments of ExtraConstraint must both be subtypes of
                    nk.hilbert.constraint.DiscreteHilbertConstraint, but one or more are not.

                    To verify if those are valid arguments you can check it with
                    isinstance(arg, nk.hilbert.constraint.DiscreteHilbertConstraint).
                    """
                )
            )

        return type(base_constraint), type(extra_constraint)

    # ------ End of Parametric class definition
