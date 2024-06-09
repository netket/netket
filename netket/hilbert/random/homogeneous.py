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


from netket.hilbert import HomogeneousHilbert
from netket.utils.dispatch import dispatch


@dispatch
def random_state(hilb: HomogeneousHilbert, key, batches: int, *, dtype=None):
    return random_state(hilb, hilb.constraint, key, batches, dtype=dtype)


# TODO: add a hilb.local_indices_to_states to have a generic
# implementation
#
# @dispatch
# @partial(jax.jit, static_argnames=("hilb", "batches", "dtype"))
# def random_state(  # noqa: F811
#     hilb: HomogeneousHilbert, constraint: None, key, batches: int, *, dtype=None
# ):
#     if dtype is None:
#         dtype = hilb._local_states.dtype
#
#     x_ids = jax.random.randint(
#         key, shape=(batches, hilb.size), minval=0, maxval=len(hilb._local_states)
#    )
#    return hilb.local_indices_to_states(x_ids)


@dispatch
def random_state(  # noqa: F811
    hilb: HomogeneousHilbert, constraint, key, batches: int, *, dtype=None
):
    raise NotImplementedError(
        f"""
        You are using a custom constraint. You must define how to generate random states
        for this particular state.

        To do this, define

        @nk.hilbert.random.random_state.dispatch
        def random_state(hilb: MyHilbType, key, batches: int, constraint: MyConstraintType, *, dtype=None):
            return ...

        where MyHilbType is {type(hilb)} and MyConstraintType is the type of the constraint you are using.
        """
    )
