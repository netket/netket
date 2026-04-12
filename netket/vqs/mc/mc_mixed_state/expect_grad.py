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

from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket.errors import MCMixedStateExpectAndGradOnPhysicalOperatorError
from netket.operator import DiscreteOperator, DiscreteJaxOperator
from netket.vqs import expect_and_grad

from .state import MCMixedState


@expect_and_grad.dispatch
def expect_and_grad_operator_fallback(  # noqa: F811
    vstate: MCMixedState,
    operator: DiscreteOperator,
    **kwargs,
):
    raise MCMixedStateExpectAndGradOnPhysicalOperatorError(vstate, operator)


@expect_and_grad.dispatch
def expect_and_grad_operator_fallback(  # noqa: F811
    vstate: MCMixedState,
    operator: DiscreteJaxOperator,
    **kwargs,
):
    raise MCMixedStateExpectAndGradOnPhysicalOperatorError(vstate, operator)


@expect_and_grad.dispatch
def expect_and_grad_operator_fallback(  # noqa: F811
    vstate: MCMixedState,
    operator: DiscreteOperator,
    chunk_size: int | None,
    *args,
    mutable: CollectionFilter = False,
    **kwargs,
):
    raise MCMixedStateExpectAndGradOnPhysicalOperatorError(vstate, operator)


@expect_and_grad.dispatch
def expect_and_grad_operator_fallback(  # noqa: F811
    vstate: MCMixedState,
    operator: DiscreteJaxOperator,
    chunk_size: int | None,
    *args,
    mutable: CollectionFilter = False,
    **kwargs,
):
    raise MCMixedStateExpectAndGradOnPhysicalOperatorError(vstate, operator)
