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

from functools import partial
import warnings

import jax
import jax.numpy as jnp

from netket import config
from netket.errors import UnoptimisedCustomConstraintRandomStateMethodWarning
from netket.hilbert import HomogeneousHilbert
from netket.utils.dispatch import dispatch
from netket.hilbert.constraint import SumConstraint, SumOnPartitionConstraint

from .fock import _random_states_with_constraint_fock


@dispatch
def random_state(hilb: HomogeneousHilbert, key, batches: int, *, dtype=None):
    return random_state(hilb, hilb.constraint, key, batches, dtype=dtype)


@dispatch
@partial(jax.jit, static_argnames=("hilb", "batches", "dtype"))
def random_state(  # noqa: F811
    hilb: HomogeneousHilbert, constraint: None, key, batches: int, *, dtype=None
):
    if dtype is None:
        dtype = hilb._local_states.dtype

    x_ids = jax.random.randint(
        key, shape=(batches, hilb.size), minval=0, maxval=len(hilb._local_states)
    )
    res = hilb.local_indices_to_states(x_ids, dtype=dtype)
    return res


@dispatch
@partial(jax.jit, static_argnames=("hilb", "batches", "dtype"))
def random_state(  # noqa: F811
    hilb: HomogeneousHilbert,
    constraint: SumConstraint,
    key,
    batches: int,
    *,
    dtype=None,
):
    local_states = hilb._local_states
    if dtype is None:
        dtype = hilb._local_states.dtype

    # Convert total constraint to Fock-like total number of excitations
    n_excitations = (
        constraint.sum_value - (local_states.start * hilb.size)
    ) // local_states.step

    samples_indx = _random_states_with_constraint_fock(
        n_excitations, hilb.shape, key, (batches,), dtype
    )
    return hilb.local_indices_to_states(samples_indx, dtype=dtype)


@dispatch
@partial(jax.jit, static_argnames=("hilb", "batches", "dtype"))
def random_state(  # noqa: F811
    hilb: HomogeneousHilbert,
    constraint: SumOnPartitionConstraint,
    key,
    batches: int,
    *,
    dtype=None,
):
    if dtype is None:
        dtype = hilb._local_states.dtype

    # Convert total constraint to Fock-like total number of excitations
    local_states = hilb._local_states
    n_excitations = [
        (sum_val - (local_states.start * hilb.size)) // local_states.step
        for sum_val in constraint.sum_values
    ]

    n_subspaces = len(n_excitations)
    keys = jax.random.split(key, len(n_excitations))
    shape = hilb.shape
    sizes = (0,) + constraint.sizes

    # Generate a valid configuration for every sub-partition
    vs = [
        _random_states_with_constraint_fock(
            n_excitations[i],
            shape[sum(sizes[: i + 1]) : sum(sizes[: i + 2])],
            keys[i],
            (batches,),
            dtype,
        )
        for i in range(n_subspaces)
    ]
    # then concatenate
    samples_indx = jnp.concatenate(vs, axis=-1)

    return hilb.local_indices_to_states(samples_indx, dtype=dtype)


@dispatch
@partial(jax.jit, static_argnames=("hilb", "batches", "dtype"))
def random_state(  # noqa: F811
    hilb: HomogeneousHilbert, constraint, key, batches: int, *, dtype=None
):
    if config.netket_random_state_fallback_warning:
        warnings.warn(
            UnoptimisedCustomConstraintRandomStateMethodWarning(hilb, constraint)
        )

    keys = jax.random.split(key, batches + 1)
    states = random_state(hilb, None, keys[0], batches, dtype=dtype)

    def _loop_until_ok(state, key):
        def __body(args):
            state, _key = args
            _key, subkey = jax.random.split(_key)
            new_state = random_state(hilb, None, subkey, 1, dtype=dtype)[0]
            return (new_state, _key)

        def __cond(args):
            state, _ = args
            return jnp.logical_not(constraint(state))

        return jax.lax.while_loop(__cond, __body, (state, key))[0]

    return jax.vmap(_loop_until_ok, in_axes=(0, 0))(states, keys[1:])


@dispatch
def flip_state_scalar(hilb: HomogeneousHilbert, key, σ, idx):  # noqa: F811
    local_dimension = len(hilb._local_states)

    if local_dimension < 2:
        return σ, σ[idx]

    # Get site to flip, convert that individual site to indices
    σi_old = σ[idx]
    xi_old = hilb.states_to_local_indices(σi_old)

    if local_dimension == 2:
        # hardcode for 2-dim, where there is no randomness
        xi_new = jnp.where(xi_old == 1, 0, 1).astype(xi_old.dtype)
    else:
        # compute flipped index
        r = jax.random.uniform(key)
        xi_new = jax.numpy.floor(r * (local_dimension - 1))
        xi_new = xi_new + (xi_new >= xi_old)
        xi_new = xi_new.astype(xi_old.dtype)

    # return
    σ_new = σ.at[idx].set(hilb.local_indices_to_states(xi_new, dtype=σ.dtype))

    return σ_new, σi_old
