# Copyright 2026 The NetKet Authors - All rights reserved.
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

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from netket._src.stats.local_estimators import LocalEstimatorsBatch
from netket.operator import AbstractOperator
from netket.operator._abstract_observable import AbstractObservable
from netket.stats import Stats
from netket.utils.numbers import is_scalar
from netket.vqs import (
    FullSumState,
    MCState,
    expect,
    get_local_kernel,
    get_local_kernel_arguments,
)
from netket.vqs.mc.common import local_estimators


def _vscore_combinator(trace_diagonal: float):
    def combinator(mu):
        return (mu[1] - mu[0] ** 2) / (mu[0] - trace_diagonal) ** 2

    return combinator


class VScore(AbstractObservable):
    r"""
    Observable computing the V-score of a quantum operator :math:`H`:

    .. math::

        V_{\mathrm{score}} =
        \frac{\mathrm{Var}(H)}{(\langle H \rangle - \mathrm{trace\_diagonal})^2}
        =
        \frac{\langle H^2 \rangle - \langle H \rangle^2}
        {(\langle H \rangle - \mathrm{trace\_diagonal})^2}.

    The ``trace_diagonal`` parameter is a mandatory diagonal-energy shift used in
    the denominator.
    """

    def __init__(
        self,
        operator: AbstractOperator,
        *,
        trace_diagonal: float,
    ):
        super().__init__(operator.hilbert)

        trace_diagonal = jnp.asarray(trace_diagonal)
        if (not is_scalar(trace_diagonal)) or jnp.iscomplexobj(trace_diagonal):
            raise TypeError("`trace_diagonal` should be a real scalar number.")

        self._operator = operator
        self._operator_squared = operator @ operator
        self._trace_diagonal = float(trace_diagonal)

    @property
    def operator(self) -> AbstractOperator:
        return self._operator

    @property
    def operator_squared(self) -> AbstractOperator:
        return self._operator_squared

    @property
    def trace_diagonal(self) -> float:
        return self._trace_diagonal

    def __repr__(self):
        return f"VScore(op={self.operator}, trace_diagonal={self.trace_diagonal})"


@local_estimators.dispatch
def vscore_local_estimators(
    vstate: MCState, vscore_op: VScore, chunk_size: int | None
) -> LocalEstimatorsBatch:  # noqa: F811
    if vscore_op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    local_kernel = get_local_kernel(vstate, vscore_op.operator, chunk_size)
    local_kernel2 = get_local_kernel(vstate, vscore_op.operator_squared, chunk_size)

    sigma, args = get_local_kernel_arguments(vstate, vscore_op.operator)
    sigma, args2 = get_local_kernel_arguments(vstate, vscore_op.operator_squared)

    n_chains = sigma.shape[0]
    if jnp.ndim(sigma) != 2:
        sigma = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(
            sigma, 0, sigma.ndim - 1
        )

    if chunk_size is not None:
        local_kernel = partial(local_kernel, chunk_size=chunk_size)
        local_kernel2 = partial(local_kernel2, chunk_size=chunk_size)

    W = vstate.variables
    O_loc = local_kernel(vstate._apply_fun, W, sigma, args).real
    O2_loc = local_kernel2(vstate._apply_fun, W, sigma, args2).real

    data = jnp.stack([O_loc, O2_loc], axis=-1).reshape(n_chains, -1, 2)
    return LocalEstimatorsBatch(
        data=data,
        combinator=_vscore_combinator(vscore_op.trace_diagonal),
    )


@expect.dispatch
def expect(vstate: MCState, vscore_op: VScore, chunk_size: int | None):  # noqa: F811
    return local_estimators(vstate, vscore_op, chunk_size).to_stats()


@expect.dispatch
def expect(vstate: FullSumState, vscore_op: VScore):  # noqa: F811
    if vscore_op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    operator_mtrx = vscore_op.operator.to_dense()
    operator_squared_mtrx = vscore_op.operator_squared.to_dense()

    return _expect_vscore_fs(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        vstate._all_states,
        operator_mtrx,
        operator_squared_mtrx,
        vscore_op.trace_diagonal,
    )


@partial(jax.jit, static_argnames=("afun",))
def _expect_vscore_fs(
    afun,
    params,
    model_state,
    sigma,
    operator_mtrx,
    operator_squared_mtrx,
    trace_diagonal,
):
    W = {"params": params, **model_state}

    state = jnp.exp(afun(W, sigma))
    state = state / jnp.linalg.norm(state)

    E_mean = (state.conj() @ (operator_mtrx @ state)).real
    E2_mean = (state.conj() @ (operator_squared_mtrx @ state)).real
    vscore = (E2_mean - E_mean**2) / (E_mean - trace_diagonal) ** 2

    return Stats(mean=vscore, error_of_mean=0.0, variance=0.0)
