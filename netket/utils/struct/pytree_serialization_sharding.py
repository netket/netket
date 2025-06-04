from typing import Protocol
import dataclasses
from functools import partial

import numpy as np

import warnings
import jax

from flax import serialization


# Specify the signature for the functions used to specify the deserialization logic
class ShardedDeserializationFunction(Protocol):
    def __call__(
        self, value_target: jax.Array, value_state: jax.Array, *, name: str = "."
    ) -> jax.Array: ...


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

    deserialization_function: ShardedDeserializationFunction | str | None = "relaxed"
    """
    Function to use to deserialize the data. Can be a callable with the signature:

    .. code-block:: python

        def f(value_target: jax.Array, value_state: jax.Array, *, name: str = ".") -> jax.Array

    or one of the following strings:

    - **"fail"**: Raise an error if the sharded data does not match the target data.
    - **"strict"**: Raise an error if the sharded data does not match the target data.
    - **"relaxed"**: Ignore extra data in the sharded data if the target data is smaller
      than the serialized data; error if the target data is larger.
    - **"relaxed-ignore-errors"**: Ignore extra data in the sharded data if the target data is
      smaller, and do nothing if the target data is larger.
    - **"relaxed-rng-key"**: Special case for RNG keys, where we can safely truncate the
      serialized data if the target data is larger.

    The default is **"relaxed"**.
    """

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
            elif self.deserialization_function == "relaxed-rng-key":
                setattr(
                    self,
                    "deserialization_function",
                    from_flax_state_dict_sharding_relaxed_rng_key,
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
    # This exists for backward compatibility...
    from netket.jax.sharding import gather
    from netket import config as nkconfig

    if (
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
    unsharded_update = serialization.from_state_dict(
        value_target, value_state, name=name
    )
    result = jax.lax.with_sharding_constraint(unsharded_update, value_target.sharding)
    return result


def from_flax_state_dict_sharding_relaxed(
    value_target, value_state, *, name=".", ignore_errors: bool = False
):
    # sharding imports
    sharded_dim = 0

    serialized_total_sharded_size = (
        value_state.shape[sharded_dim] if value_state.ndim > 0 else 0
    )
    target_total_sharded_size = (
        value_target.shape[sharded_dim] if value_target.ndim > 0 else 1
    )

    if serialized_total_sharded_size > target_total_sharded_size:
        if jax.process_index() == 0:
            warnings.warn(
                f"Sharded data ({name}) has shape {value_state.shape} but target has shape {value_target.shape}."
                "This probably means that you serialized the data with more nodes. We will try to load it anyway "
                "by ignoring the extra data."
            )
        if value_target.ndim == 0:
            value_state = value_state[:1]
        else:
            value_state = value_state[: value_target.shape[sharded_dim]]
    elif serialized_total_sharded_size < target_total_sharded_size:
        if ignore_errors:
            return value_target
        raise ValueError(
            f"\n"
            f"Sharded data for field '{name}' size mismatch: \n"
            f"\t Info                                          \n"
            f"\t       sharded axis #           : {sharded_dim}\n"
            f"\t       # jax processes          : {jax.process_count()}\n"
            f"\t serialized shape   (global)    : {value_state.shape}\n"
            f"\t           (total_sharded_size) : {serialized_total_sharded_size}\n"
            f"\t restore trgt shape (global)    : {value_target.shape}\n"
            f"\t           (total_sharded_size) : {target_total_sharded_size}\n"
            "This probably means that you serialized the data with less nodes. We cannot load the data in this case."
            "\n"
        )
    unsharded_update = serialization.from_state_dict(
        value_target, value_state, name=name
    )
    result = jax.lax.with_sharding_constraint(unsharded_update, value_target.sharding)
    return result


def from_flax_state_dict_sharding_relaxed_rng_key(
    value_target, value_state, *, name=".", ignore_errors: bool = False
):
    """
    Equivalent to `from_flax_state_dict_sharding_relaxed`, but special cased
    for rng keys, which we know that can be safely truncated if the target state
    is larger than the serialized state.
    """
    # sharding imports
    sharded_dim = 0

    serialized_total_sharded_size = (
        value_state.shape[sharded_dim] if value_state.ndim > 0 else 0
    )
    target_total_sharded_size = (
        value_target.shape[sharded_dim] if value_target.ndim > 0 else 1
    )

    if serialized_total_sharded_size > target_total_sharded_size:
        if jax.process_index() == 0:
            warnings.warn(
                f"""
                Serialized RNG key data ({name}) has a BIGGER shape {value_state.shape} that the
                target to be restored to ({value_target.shape}).

                We will truncate the extra data.

                The loading will succeed, but the random numbers generated will be different
                from the original ones (this should usually not be a problem).
                """
            )
        if value_target.ndim == 0:
            value_state = value_state[:1]
        else:
            value_state = value_state[: value_target.shape[sharded_dim]]
    elif serialized_total_sharded_size < target_total_sharded_size:
        if jax.process_index() == 0:
            warnings.warn(
                f"""
                Serialized RNG key data ({name}) has a SMALLER shape {value_state.shape} that the
                target to be restored to ({value_target.shape}).

                We will use the available data to construct a new, larger RNG key.

                The loading SHOULD succeed, but the random numbers generated will be different
                from the original ones (this should usually not be a problem).
                If it fails, open an issue.
                """
            )
        target_shape = list(value_target.shape)
        target_shape[sharded_dim] = value_target.shape[sharded_dim]
        target_shape = tuple(target_shape)

        # Generate a new RNG key data with the target shape, using the current rng data as
        # a seed
        rng = np.random.default_rng(value_state)
        new_value_state = rng.integers(
            0,
            high=np.iinfo(value_state.dtype).max,
            dtype=value_state.dtype,
            size=target_shape,
        )
        # Copy the data from the serialized state to the new state
        new_value_state[: value_state.shape[0]] = value_state
        value_state = new_value_state

    unsharded_update = serialization.from_state_dict(
        value_target, value_state, name=name
    )
    result = jax.lax.with_sharding_constraint(unsharded_update, value_target.sharding)
    return result
