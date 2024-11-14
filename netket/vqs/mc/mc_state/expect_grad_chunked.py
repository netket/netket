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

from typing import Any
import warnings

from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket.vqs import expect_and_grad

# TODO:  merged with above once stabilised
from netket.operator._abstract_observable import AbstractObservable

from .state import MCState
from .expect_grad import expect_and_grad_nonhermitian


def ignore_chunk_warning(vstate, operator, chunk_size, name=""):
    return f"""
            Ignoring chunk_size={chunk_size} for {name} method with signature
            ({type(vstate)}, {type(operator)}) because no implementation supporting
            chunking for this signature exists.
            """


# If chunk size is unspecified, set it to None
@expect_and_grad.dispatch
def expect_and_grad_chunking_unspecified(  # noqa: F811
    vstate: MCState,
    operator: AbstractObservable,
    **kwargs,
):
    return expect_and_grad(vstate, operator, None, **kwargs)


# if no implementation exists for batched, run the code unbatched
@expect_and_grad.dispatch(precedence=-10)
def expect_and_grad_fallback(  # noqa: F811
    vstate: MCState,
    operator: AbstractObservable,
    chunk_size: int | tuple,
    *args,
    **kwargs,
):
    warnings.warn(
        ignore_chunk_warning(vstate, operator, chunk_size, name="expect_and_grad")
    )
    return expect_and_grad(vstate, operator, None, *args, **kwargs)


@expect_and_grad_nonhermitian.dispatch(precedence=-10)
def expect_and_grad_nonhermitian_chunk_fallback(
    vstate: MCState,
    Ô,
    chunk_size: Any,
    **kwargs,
):
    warnings.warn(
        ignore_chunk_warning(vstate, Ô, chunk_size, name="expect_and_grad_nonhermitian")
    )
    return expect_and_grad_nonhermitian(vstate, Ô, None, **kwargs)
