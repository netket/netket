import jax.numpy as jnp

from netket.utils.types import Array
from netket.utils import struct

from .base import (
    optimalConstrainedHilbertindex,
    ConstrainedHilbertIndex,
    DiscreteHilbertConstraint,
)


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

    def __call__(self, x: Array) -> Array:
        conditions = jnp.stack(
            [self.base_constraint(x), self.extra_constraint(x)], axis=0
        )
        return jnp.all(conditions, axis=0)


@optimalConstrainedHilbertindex.dispatch
def optimalConstrainedHilbertindex(local_states, size, constraint: ExtraConstraint):
    # If we have a constraint, we tentatively construct a specialised Hilbert index for that particular constraint.
    # If this specialised indexer object exists, we check whether it is more efficient than the generic
    # ConstrainedHilbertIndex one. If it is more efficient, we use it, otherwise we keep the generic one.
    bare_index = optimalConstrainedHilbertindex(
        local_states, size, constraint.base_constraint
    )
    return ConstrainedHilbertIndex(bare_index, constraint.extra_constraint)
