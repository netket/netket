from typing import Protocol
import dataclasses
from functools import partial

import warnings
import jax

from flax import serialization


# Specify the signature for the functions used to specify the deserialization logic
class ShardedDeserializationFunction(Protocol):
    def __call__(
        self, value_target: jax.Array, value_state: jax.Array, *, name: str = "."
    ) -> jax.Array:
        ...


@dataclasses.dataclass
class ShardedFieldSpec:
    """Specification of a sharded field.

    Used to specify (for the time being) how to handle serialization/deserialization (using
    flax.serialize) of a field in a NetKet-style Pytree.
    """

    sharded: bool = True
    """Boolean indicating whether the field is sharded or not. If False, all other
    fields are ignored.
    """

    mpi_sharded_axis: int | None = 0
    """
    Optional integer indicating which axis is to be considered sharded when running with
    MPI. Defaults to 0.

    If this is specified, loading with MPI a state saved from a run with jax sharding will
    scatter/partition the data along the specified axis.

    If None, MPI is not supported.
    """

    deserialization_function: ShardedDeserializationFunction | str | None = "relaxed"

    def __post_init__(self):
        if isinstance(self.deserialization_function, type(None)):
            setattr(self, "deserialization_function", "fail")
        if isinstance(self.deserialization_function, str):
            if self.deserialization_function == "fail":
                setattr(
                    self, "deserialization_function", from_flax_state_dict_sharding_fail
                )
            elif self.deserialization_function == "strict":
                setattr(
                    self,
                    "deserialization_function",
                    from_flax_state_dict_sharding_strict,
                )
            elif self.deserialization_function == "relaxed":
                setattr(
                    self,
                    "deserialization_function",
                    from_flax_state_dict_sharding_relaxed,
                )
            elif self.deserialization_function == "relaxed-ignore-errors":
                setattr(
                    self,
                    "deserialization_function",
                    partial(from_flax_state_dict_sharding_relaxed, ignore_errors=True),
                )
            else:
                raise ValueError(
                    f"Unknown deserialization function logic {self.deserialization_function}."
                    "Valid values are 'fail/strict/relaxed' or a custom callable."
                )

    def __bool__(self):
        # Make this behave as a boolean, so we use it as sharding specification
        return self.sharded


def to_flax_state_dict_sharding(sharded_data):
    from netket.utils import mpi

    # This exists for backward compatibility...
    from netket.jax.sharding import gather
    from netket import config as nkconfig

    if mpi.n_nodes > 1:
        gathered_data = mpi.mpi_gather(sharded_data)
        result = gathered_data.reshape((-1,) + gathered_data.shape[2:])
    elif (
        nkconfig.netket_experimental_sharding
        and not nkconfig.netket_experimental_sharding_fast_serialization
    ):
        # 'old' behaviour (compatibility for now... disabled with the)
        # flag above
        result = gather(sharded_data)
    else:
        result = sharded_data

    return result


# Fail mode
def from_flax_state_dict_sharding_fail(value_target, value_state, *, name="."):
    raise RuntimeError(
        f"Deserialization for this sharded field [{name}] was disabled by "
        "the deserialization specification of the Pytree."
    )


# Strict mode
def from_flax_state_dict_sharding_strict(value_target, value_state, *, name="."):
    # sharding imports
    from netket.utils import mpi

    sharded_dim = 0

    if mpi.n_nodes > 1:
        # Assuming every rank got the full data... So we don't need to broadcast
        if (
            value_state.shape[sharded_dim]
            != mpi.n_nodes * value_target.shape[sharded_dim]
        ):
            raise ValueError(
                f"Sharded data ({name}) has shape {value_state.shape} but target has shape {value_target.shape}."
                "This probably means that you serialized the data with a different number of nodes or a different "
                "array size alltogether."
            )
        scattered_value_state = value_state.reshape(
            (mpi.n_nodes,) + value_target.shape
        )[mpi.rank]
        result = serialization.from_state_dict(
            value_target, scattered_value_state, name=name
        )
        result = jax.lax.with_sharding_constraint(result, value_target.sharding)

    else:  # if processes > 1
        unsharded_update = serialization.from_state_dict(
            value_target, value_state, name=name
        )
        result = jax.lax.with_sharding_constraint(
            unsharded_update, value_target.sharding
        )
    return result


def from_flax_state_dict_sharding_relaxed(
    value_target, value_state, *, name=".", ignore_errors: bool = False
):
    # sharding imports
    from netket.utils import mpi

    sharded_dim = 0

    serialized_shape_size = (
        value_state.shape[sharded_dim] if value_state.ndim > 0 else 0
    )
    target_shape_size = (
        mpi.n_nodes * value_target.shape[sharded_dim]
        if value_target.ndim > 0
        else mpi.n_nodes
    )

    if serialized_shape_size > target_shape_size:
        if mpi.rank == 0 and jax.process_index() == 0:
            warnings.warn(
                f"Sharded data ({name}) has shape {value_state.shape} but target has shape {value_target.shape}."
                "This probably means that you serialized the data with more nodes. We will try to load it anyway "
                "by ignoring the extra data."
            )
        if value_target.ndim == 0:
            value_state = value_state[: mpi.n_nodes]
        else:
            value_state = value_state[: mpi.n_nodes * value_target.shape[sharded_dim]]
    elif serialized_shape_size < target_shape_size:
        if ignore_errors:
            return value_target
        raise ValueError(
            f"Sharded data ({name}) has shape {value_state.shape} but target has shape ({mpi.n_nodes}*){value_target.shape}."
            "This probably means that you serialized the data with less nodes. We cannot load the data in this case."
        )

    if mpi.n_nodes > 1:
        # Assuming every rank got the full data... So we don't need to broadcast
        scattered_value_state = value_state.reshape(
            (mpi.n_nodes,) + value_target.shape
        )[mpi.rank]
        result = serialization.from_state_dict(
            value_target, scattered_value_state, name=name
        )
        result = jax.lax.with_sharding_constraint(result, value_target.sharding)

    else:  # if processes > 1
        unsharded_update = serialization.from_state_dict(
            value_target, value_state, name=name
        )
        result = jax.lax.with_sharding_constraint(
            unsharded_update, value_target.sharding
        )
    return result
