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

from netket.hilbert import Spin
from netket.utils.dispatch import dispatch


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
