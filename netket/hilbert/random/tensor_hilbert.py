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

from netket.hilbert import TensorHilbert
from netket.utils.dispatch import dispatch

from netket.utils.samples_pytree import SampleWrapperExample


@dispatch
def random_state(hilb: TensorHilbert, key, batches: int, *, dtype):
    keys = jax.random.split(key, hilb._n_hilbert_spaces)

    vs = tuple(
        random_state(hilb._hilbert_spaces[i], keys[i], batches, dtype=dtype)
        for i in range(hilb._n_hilbert_spaces)
    )
    if isinstance(batches, int):
        batches = (batches,)
    structure = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape[len(batches) :], x.dtype), vs
    )
    return SampleWrapperExample(vs, structure)


def _make_subfun(hilb, i, sub_hi):
    def subfun(args):
        key, state, index = args

        sub_state = state.sub_states[i]
        # TODO can't we just compute the random idx locally?
        new_sub_state, old_val = flip_state_scalar(
            sub_hi, key, sub_state, index - hilb._cum_indices[i]
        )
        new_state = state.replace(
            sub_states=state.sub_states[:i]
            + (new_sub_state,)
            + state.sub_states[i + 1 :]
        )
        return new_state, old_val

    return subfun


@dispatch
def flip_state_scalar(hilb: TensorHilbert, key, state, index):
    subfuns = []
    for i, sub_hi in enumerate(hilb._hilbert_spaces):
        subfuns.append(_make_subfun(hilb, i, sub_hi))

    branches = []
    for i in hilb._hilbert_i:
        branches.append(subfuns[i])

    # vmap converts this to select, so we could just compute all rules here
    # and select directly
    return jax.lax.switch(index, branches, (key, state, index))
