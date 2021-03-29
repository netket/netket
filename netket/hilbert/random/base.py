# Copyright 2021 The NetKet Authors - All rights reserved.
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

from functools import partial, singledispatch

import numpy as np
import jax


def random_state(hilb, key, size=None, dtype=np.float32):
    r"""Generates either a single or a batch of uniformly distributed random states.

    Args:
        size: If provided, returns a batch of configurations of the form (size, #) if size
            is an integer or (*size, #) if it is a tuple and where # is the Hilbert space size.
            By default, a single random configuration with shape (#,) is returned.
        out: If provided, the random quantum numbers will be inserted into this array,
             which should be of the appropriate shape (see `size`) and data type.
        rgen: The random number generator. If None, the global NetKet random
            number generator is used.

    Example:
        >>> hi = netket.hilbert.Qubit(N=2)
        >>> hi.random_state()
        array([0., 1.])
        >>> hi.random_state(size=2)
        array([[0., 0.], [1., 0.]])
    """
    if size is None:
        return random_state_scalar(hilb, key, dtype)
    else:
        return random_state_batch(hilb, key, size, dtype)


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


##############################
### Random_state functions ###
##############################


@singledispatch
def random_state_scalar(hilb, key, dtype):
    """
    Generates a single random state-vector given an hilbert space and a rng key.
    """
    # Attempt to use the scalar method
    raise NotImplementedError(
        f"""
                              random_state_scalar(hilb, key, dtype) is not implemented
                              for hilbert space of type {type(hilb)}. 

                              See the documentation of 
                              nk.hilbert.random.register_random_state_impl.
                              """
    )


@singledispatch
def random_state_batch(hilb, key, size, dtype):
    """
    Generates a batch of random state-vectors given an hilbert space and a rng key.
    """
    # Attempt to use the batch method
    raise NotImplementedError(
        f"""
                              random_state_batch(hilb, key, size, dtype) is not implemented
                              for hilbert space of type {type(hilb)}. 

                              See the documentation of 
                              nk.hilbert.random.register_random_state_impl.
                              """
    )


def _random_state_scalar_default_impl(hilb, key, dtype, batch_rule):
    return batch_rule(hilb, key, 1, dtype).reshape(-1)


def _random_state_batch_default_impl(hilb, key, size, dtype, scalar_rule):
    keys = jax.random.split(key, size)
    res = jax.vmap(scalar_rule, in_axes=(None, 0, None), out_axes=0)(hilb, key, dtype)
    return res


def register_random_state_impl(clz=None, *, scalar=None, batch=None):
    """
    Register an implementation for the function generating random
    state for the given Hilbert space class.

    The rule can be implemented both as a scalar rule and as a batched
    rule, but the best performance will be obtained by implementing
    the batched version.

    The missing rule will be auto-implemented from the over.

    scalar must have signature
        (hilb, key, dtype) -> vector
    batch must have signature
        (hilb, key, size, dtype) -> matrix of states

    The function will be jit compiled, so make sure to use jax.numpy.
    Hilbert is passed as a static object.

    Arguments:
        clz: The class of the hilbert space
        scalar: The function computing a single random state
        batch: the function computing batches of random states
    """
    if scalar is None and batch is None:
        raise ValueError("You must at least provide a scalar or batch rule.")

    scalar_rule = scalar
    batch_rule = batch

    if scalar is None:
        if clz is None:
            clz = list(batch.__annotations__.items())[0]
        scalar_rule = partial(_random_state_scalar_default_impl, batch_rule=batch_rule)

    if batch is None:
        if clz is None:
            clz = list(scalar.__annotations__.items())[0]

        batch_rule = partial(_random_state_batch_default_impl, scalar_rule=scalar_rule)

    random_state_scalar.register(clz, scalar_rule)
    random_state_batch.register(clz, batch_rule)


##############################
### flip_state functions ###
##############################


@singledispatch
def flip_state_scalar(hilb, key, state, indx):
    raise NotImplementedError(
        f"""
                              flip_state_scalar(hilb, key, state, indx) is not implemented
                              for hilbert space of type {type(hilb)}. 

                              See the documentation of 
                              nk.hilbert.random.register_flip_state_impl
                              """
    )


@singledispatch
def flip_state_batch(hilb, key, states, indxs):
    raise NotImplementedError(
        f"""
                              flip_state_batch(hilb, key, states, indx) is not implemented
                              for hilbert space of type {type(hilb)}. 

                              See the documentation of 
                              nk.hilbert.random.register_flip_state_impl
                              """
    )


def _flip_state_scalar_default_impl(hilb, key, state, indx, batch_rule):
    new_state, old_val = batch_rule(
        hilb, key, state.reshape((1, -1)), indx.reshape(1, -1)
    )
    return new_state.reshape(-1), old_val.reshape(())


def _flip_state_batch_default_impl(hilb, key, states, indxs, scalar_rule):
    keys = jax.random.split(key, states.shape[0])
    res = jax.vmap(scalar_rule, in_axes=(None, 0, 0, 0), out_axes=0)(
        hilb, keys, states, indxs
    )
    return res


def register_flip_state_impl(clz=None, *, scalar=None, batch=None):
    """
    Register an implementation for the function generating and
    applying random local states for the given Hilbert space class.

    The rule can be implemented both as a scalar rule and as a batched
    rule, but the best performance will be obtained by implementing
    the batched version.

    The missing rule will be auto-implemented from the over.

    scalar must have signature
        (hilb, key, state, indx) -> (new state, state[indx])
    batch must have signature
        (hilb, key, states, indxs) -> batch of scalar results

    The function will be jit compiled, so make sure to use jax.numpy.
    Hilbert is passed as a static object.

    Arguments:
        clz: The class of the hilbert space
        scalar: The function computing a single entry
        batch: the function computing batches
    """
    if scalar is None and batch is None:
        raise ValueError("You must at least provide a scalar or batch rule.")

    scalar_rule = scalar
    batch_rule = batch

    if scalar is None:
        if clz is None:
            clz = list(batch.__annotations__.items())[0]
        scalar_rule = partial(_flip_state_scalar_default_impl, batch_rule=batch_rule)

    if batch is None:
        if clz is None:
            clz = list(scalar.__annotations__.items())[0]

        batch_rule = partial(_flip_state_batch_default_impl, scalar_rule=scalar_rule)

    flip_state_scalar.register(clz, scalar_rule)
    flip_state_batch.register(clz, batch_rule)
