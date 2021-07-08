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

from textwrap import dedent
from typing import Tuple, Union

import jax
import numpy as np

from netket.utils.dispatch import dispatch

Dim = Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]]


@dispatch
def random_state(hilb, key, *, size=None, dtype=np.float32):
    r"""Generates either a single or a batch of uniformly distributed random states.

    Args:
        hilb: The hilbert space
        key: The Jax PRNGKey
        size: If provided, returns a batch of configurations of the form (size, #) if
            size is an integer or (*size, #) if it is a tuple and where # is the Hilbert
            space size. By default, a single random configuration with shape (#,) is
            returned.
        dtype: The dtype of the resulting states.

    Example:

        >>> import netket, jax
        >>> hi = netket.hilbert.Qubit(N=2)
        >>> print(hi.random_state(key=jax.random.PRNGKey(0)))
        [1. 0.]
        >>> print(hi.random_state(size=2, key=jax.random.PRNGKey(1)))
        [[0. 1.]
         [0. 0.]]
    """
    return random_state(hilb, key, size, dtype=dtype)


@dispatch
def random_state(hilb, key, size, dtype):  # noqa: F811
    return random_state(hilb, key, size, dtype=dtype)


@dispatch
def random_state(hilb, key, size: None, *, dtype):  # noqa: F811
    return random_state(hilb, key, 1, dtype=dtype)[0]


@dispatch
def random_state(hilb, key, size: Dim, *, dtype):  # noqa: F811
    n = int(np.prod(size))
    return random_state(hilb, key, n, dtype=dtype).reshape(*size, -1)


@dispatch
def random_state(hilb, key, size: int, *, dtype):  # noqa: F811
    raise NotImplementedError(
        dedent(
            f"""
            random_state(hilb, key, size : int, *, dtype) is not implemented for the
            hilbert space {type(hilb)}.

            Define the above function as follows:

            >>>from netket.utils.dispatch import dispatch
            >>>@dispatch
            >>>def random_state(hilb : {type(hilb)}, key, size : int, *, dtype):
            >>>  ...
            >>>  return batch_of_size_states
        """
        )
    )


@dispatch
def random_state(hilb, key, size: None, *, dtype):  # noqa: F811
    return random_state(hilb, key, 1, dtype=dtype)[0]


@dispatch
def random_state(hilb, key, size: Dim, *, dtype):  # noqa: F811
    n = int(np.prod(size))
    return random_state(hilb, key, n, dtype=dtype).reshape(*size, -1)


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


@dispatch
def flip_state_scalar(hilb, key, state, indx):
    new_state, old_val = flip_state_batch(
        hilb, key, state.reshape(1, -1), indx.reshape(1, -1)
    )
    return new_state.reshape(-1), old_val.reshape()


@dispatch
def flip_state_batch(hilb, key, states, indxs):
    keys = jax.random.split(key, states.shape[0])
    res = jax.vmap(flip_state_scalar, in_axes=(None, 0, 0, 0), out_axes=0)(
        hilb, keys, states, indxs
    )
    return res
