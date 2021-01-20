from functools import partial, singledispatch

import jax


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
    res = random_state_batch_impl(hilb, key, size, dtype)
    if res is NotImplemented:
        # If it fails, vmap over the scalar method
        keys = jax.random.split(key, size)
        res = jax.vmap(random_state_scalar_impl, in_axes=(None, 0, None), out_axes=0)(
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


#####################################################################


def flip_state(hilb, key, state, indices):
    r"""
    Given a state `σ` and an index `i`, randomly flips `σ[i]` so that
    `σ_new[i] ≢ σ[i]`.

    Also accepts batched inputs, where state is a batch and indices is a
    vector of ints.

    Returns:
        new_state: a state or batch of states, with one site flipped
        old_vals: a scalar (or vector) of the old values at the flipped sites
    """
    if state.ndim == 1:
        return flip_state_scalar(hilb, key, state, indices)
    else:
        return flip_state_batch(hilb, key, state, indices)


@partial(jax.jit, static_argnums=(0,))
def flip_state_scalar(hilb, key, state, indx):
    res = flip_state_scalar_impl(hilb, key, state, indx)
    if res is NotImplemented:
        res = flip_state_batch_impl(
            hilb, key, state.reshape((1, -1)), indx.reshape(1, -1)
        )
        if res is NotImplemented:
            raise NotImplementedError(
                """_jax_random_state_scalar(hilb, key, dtype) is not defined for hilb of type {} or a supertype.
                For better performance you should define _jax_random_state_batch.""".format(
                    type(hilb)
                )
            )
        else:
            new_state, old_vals = res
            res = new_state.reshape((-1,)), olv_vals.reshape((-1,))

    return res


@partial(jax.jit, static_argnums=(0,))
def flip_state_batch(hilb, key, states, indxs):
    res = flip_state_batch_impl(hilb, key, states, indxs)
    if res is NotImplemented:
        keys = jax.random.split(key, states.shape[0])
        res = jax.vmap(flip_state_scalar_impl, in_axes=(None, 0, 0, 0), out_axes=0)(
            hilb, keys, states, indxs
        )
    return res


# Implementations must implement flip_state_scalar_impl, and, if possible
# flip_state_batch_impl for speed. If flip_state_batch_impl is
# not implemented than a default vmap over the scalar version is used.


@singledispatch
def flip_state_scalar_impl(hilb, key, state, indices):
    return NotImplemented


@singledispatch
def flip_state_batch_impl(hilb, key, state, indices):
    return NotImplemented
