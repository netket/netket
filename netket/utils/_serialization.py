import jax
import jax.numpy as jnp


def _remove_prngkey(val):
    if isinstance(val, jax.Array) and jnp.issubdtype(val.dtype, jax.dtypes.prng_key):
        return jax.random.key_data(val)
    else:
        return val


def _restore_prngkey(target_val, state_dict_val):
    if isinstance(target_val, jax.Array) and jnp.issubdtype(
        target_val.dtype, jax.dtypes.prng_key
    ):
        return jax.random.wrap_key_data(
            state_dict_val, impl=jax.random.key_impl(target_val)
        )
    else:
        return state_dict_val


def remove_prngkeys(tree):
    """
    Removes PRNG keys from a pytree, replacing them with their key_data.

    This is useful when serializing a pytree to disk, as PRNG keys are not
    serializable.

    Args:
        tree: the pytree to remove PRNG keys from.

    Returns:
        The pytree with PRNG keys replaced by their key_data.
    """
    return jax.tree.map(_remove_prngkey, tree)


def restore_prngkeys(target, state_dict):
    """
    Restores PRNG keys in a pytree from their key_data, reversing the operation
    performed by `remove_prngkeys`.

    Args:
        target: the target pytree to restore PRNG keys in.
        state_dict: the pytree containing the key_data to restore.
    """
    return jax.tree.map(_restore_prngkey, target, state_dict)
