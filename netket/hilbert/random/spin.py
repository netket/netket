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

import jax
from functools import partial

from netket.hilbert import Spin
from netket.hilbert.spin import SumConstraint
from netket.utils.dispatch import dispatch

from .fock import _random_states_with_constraint_fock


@dispatch
@partial(jax.jit, static_argnames=("hilb", "batches", "dtype"))
def random_state(  # noqa: F811
    hilb: Spin,
    constraint: SumConstraint,
    key,
    batches: int,
    *,
    dtype=None,
):
    # Generate random spin states with a given hilb._total_sz.
    # Note that this is NOT a uniform distribution over the
    # basis states of the constrained hilbert space.
    two_times_s = round(2 * hilb._s)
    two_times_total_sz = round(2 * hilb._total_sz)
    n_particles = _spin_to_fock(two_times_s * hilb.size, two_times_total_sz)
    x_fock = _random_states_with_constraint_fock(
        n_particles, hilb.shape, key, (batches,), dtype
    )
    return _fock_to_spin(two_times_s, x_fock).astype(dtype)


# For the implementations of constrained spaces, we use those found inside of
# fock.py, and simply convert the spin values (-1, 1) to fock values (0,1, ...).

_spin_to_fock = lambda two_times_s, x: (two_times_s + x) // 2
_fock_to_spin = lambda two_times_s, x: 2 * x - two_times_s


@dispatch
def flip_state_scalar(hilb: Spin, key, state, index):
    if hilb._s == 0.5:
        return _flipat_N2(key, state, index)
    else:
        return _flipat_generic(key, state, index, hilb._s)


def _flipat_N2(key, x, i):
    res = x.at[i].set(-x[i]), x[i]
    return res


def _flipat_generic(key, x, i, s):
    n_states = int(2 * s + 1)

    xi_old = x[i]
    r = jax.random.uniform(key)
    xi_new = jax.numpy.floor(r * (n_states - 1)) * 2 - (n_states - 1)
    xi_new = xi_new + 2 * (xi_new >= xi_old)
    xi_new = xi_new.astype(x.dtype)

    new_state = x.at[i].set(xi_new)
    return new_state, xi_old
