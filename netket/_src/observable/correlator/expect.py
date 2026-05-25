# Copyright 2024 The NetKet Authors - All rights reserved.
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

"""Dispatch rules for correlation function observables."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from netket.vqs import MCState, expect
from netket.vqs.mc import (
    get_local_kernel,
    get_local_kernel_arguments,
)
from netket.vqs.mc.common import local_estimators
from netket._src.stats.local_estimators import LocalEstimatorsBatch

from .correlator import ConnectedCorrelator


def _compute_local_estimator(vstate: MCState, op, chunk_size):
    kernel = get_local_kernel(vstate, op, chunk_size)
    sigma, args = get_local_kernel_arguments(vstate, op)
    sigma_flat = jax.lax.collapse(sigma, 0, sigma.ndim - 1) if sigma.ndim > 2 else sigma
    if chunk_size is not None:
        kernel = partial(kernel, chunk_size=chunk_size)
    return kernel(vstate._apply_fun, vstate.variables, sigma_flat, args)


def _stack_channels(vstate: MCState, ops, chunk_size) -> jax.Array:
    n_chains = vstate.samples.shape[0]
    channels = [_compute_local_estimator(vstate, op, chunk_size) for op in ops]
    data = jnp.stack(channels, axis=-1)
    return data.reshape(n_chains, -1, len(ops))


@local_estimators.dispatch
def connected_correlator_local_estimators(  # noqa: F811
    vstate: MCState, obs: ConnectedCorrelator, chunk_size: int | None
) -> LocalEstimatorsBatch:
    ops = [obs._op_A, obs._op_B, obs._product_op]
    data = _stack_channels(vstate, ops, chunk_size)
    return LocalEstimatorsBatch(
        data=data,
        combinator=lambda mu: mu[2] - mu[0] * mu[1],
    )


@expect.dispatch
def connected_correlator_expect(  # noqa: F811
    vstate: MCState, obs: ConnectedCorrelator, chunk_size: int | None
):
    return local_estimators(vstate, obs, chunk_size).to_stats()
