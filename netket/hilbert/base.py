import jax
import functools
from functools import partial, singledispatch


def random_state(hilb, key, size=None, dtype=jax.numpy.float32):
    """
    Generates a state or batch of random states (according to argument
    size) with specified dtype.
    """
    if size is None:
        return random_state_scalar(hilb, key, dtype)
    else:
        return random_state_batch(hilb, key, size, dtype)


@partial(jax.jit, static_argnums=(0, 2))
def random_state_scalar(hilb, key, dtype):
    """
    Generates a single random state-vector given an hilbert space and a rng key.
    """
    # Attempt to use the scalar method
    res = random_state_scalar_impl(hilb, key, dtype)
    if res is NotImplemented:
        # If the scalar method is not implemented, use the batch method and take the first batch
        res = random_state_batch_impl(hilb, key, 1, dtype).reshape(-1)
        if res is NotImplemented:
            raise NotImplementedError(
                """_jax_random_state_scalar(hilb, key, dtype) is not defined for hilb of type {} or a supertype.
                For better performance you should define _jax_random_state_batch.""".format(
                    type(hilb)
                )
            )

    return res


@partial(jax.jit, static_argnums=(0, 2, 3))
def random_state_batch(hilb, key, size, dtype):
    """
    Generates a batch of random state-vectors given an hilbert space and a rng key.
    """
    # Attempt to use the batch method
    res = _jax_random_state_batch(hilb, key, size, dtype)
    if res is NotImplemented:
        # If it fails, vmap over the scalar method
        keys = jax.random.split(key, size)
        res = jax.vmap(_jax_random_state_scalar, in_axes=(None, 0, None), out_axes=0)(
            hilb, key, dtype
        )
    return res


@singledispatch
def random_state_scalar_impl(hilb, key, dtype):
    # Implementation for jax_random_state_scalar, dispatching on the
    # type of hilbert.
    # Could probably put it in the class itself (which @singledispatch automatically
    # because of oop)?
    return NotImplemented


@singledispatch
def random_state_batch_impl(hilb, key, size, dtype):
    # Implementation for jax_random_state_batch, dispatching on the
    # type of hilbert.
    # Could probably put it in the class itself (which @singledispatch automatically
    # because of oop)?
    return NotImplemented
