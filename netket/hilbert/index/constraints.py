import jax
from flax import struct
from netket.utils.types import Scalar, Array


@struct.dataclass
class SumConstraint:
    sum_value: Scalar = struct.field(pytree_node=False)

    @jax.jit
    def __call__(self, x: Array) -> Array:
        return x.sum(axis=1) == self.sum_value
