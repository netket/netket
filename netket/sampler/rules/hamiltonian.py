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

import math

import jax
import jax.numpy as jnp

import numpy as np

from numba import jit

from netket import config
from netket.operator import DiscreteOperator, DiscreteJaxOperator
from netket.utils import struct

from .base import MetropolisRule


class HamiltonianRuleBase(MetropolisRule):
    """
    Rule proposing moves according to the terms in an operator.
    """

    # operator: AbstractOperator = struct.field(pytree_node=False / True depending on jax or not jax)
    """The (hermitian) operator giving the transition amplitudes."""

    def init_state(self, sampler, machine, params, key):
        if sampler.hilbert != self.operator.hilbert:
            raise ValueError(
                f"""
            The hilbert space of the sampler ({sampler.hilbert}) and the hilbert space
            of the operator ({self.operator.hilbert}) for HamiltonianRule must be the same.
            """
            )
        return super().init_state(sampler, machine, params, key)


@struct.dataclass
class HamiltonianRuleNumba(HamiltonianRuleBase):
    r"""
    Rule proposing moves according to the terms in an operator.

    In this case, the transition matrix is taken to be:

    .. math::

       T( \mathbf{s} \rightarrow \mathbf{s}^\\prime) = \frac{1}{\mathcal{N}(\mathbf{s})}\theta(|H_{\mathbf{s},\mathbf{s}^\prime}|),

    This rule only works on CPU! If you want to use it on GPU, you
    must use the numpy variant :class:`netket.sampler.rules.HamiltonianRuleNumpy`
    together with the numpy metropolis sampler :class:`netket.sampler.MetropolisSamplerNumpy`.
    """

    operator: DiscreteOperator = struct.field(pytree_node=False)
    """The (hermitian) operator giving the transition amplitudes."""

    def __init__(self, operator: DiscreteOperator):
        if not isinstance(operator, DiscreteOperator):
            raise TypeError(
                "Argument to HamiltonianRule must be a valid operator, "
                f"but operator is a {type(operator)}."
            )
        if config.netket_experimental_sharding:
            raise TypeError(
                "Numba-based hamiltonian sampling rule is not"
                "supported with NETKET_EXPERIMENTAL_SHARDING. To keep using"
                "sharding, use a jax based operator."
            )
        # call _setup on the operator if it exists, to warmup the cache and
        # avoid calling it in a numba callback which might break things.
        if hasattr(operator, "_setup"):
            operator._setup()
        self.operator = operator

    def transition(rule, sampler, machine, parameters, state, key, σ):
        """
        This implements the transition rule for `DiscreteOperator`s that are
        not jax-compatible by using a :ref:`jax.pure_callback`, which has a large
        overhead.

        If possible, consider using a jax-variant of the operators.
        """
        log_prob_dtype = jax.dtypes.canonicalize_dtype(float)

        def _transition(v, rand_vec):
            log_prob_corr = np.zeros((v.shape[0],), dtype=log_prob_dtype)
            v_proposed = np.empty(v.shape, dtype=v.dtype)

            sections = np.empty(v.shape[0], dtype=np.int32)
            vp, _ = rule.operator.get_conn_flattened(v, sections)

            _choose(vp, sections, rand_vec, v_proposed, log_prob_corr)

            rule.operator.n_conn(v_proposed, sections)

            log_prob_corr -= np.log(sections)
            return v_proposed, log_prob_corr

        # ideally we would pass the key to python/numba in _choose, initialise a
        # np.random.default_rng(key) and use it to generate random uniform integers.
        # However, numba does not support np states, and reseeding its MT1998 implementation
        # would be slow so we generate floats in the [0,1] range in jax and pass those
        # to python
        rand_vec = jax.random.uniform(key, shape=(σ.shape[0],))

        σp, log_prob_correction = jax.pure_callback(
            _transition,
            (
                jax.ShapeDtypeStruct(σ.shape, σ.dtype),
                jax.ShapeDtypeStruct((σ.shape[0],), log_prob_dtype),
            ),
            σ,
            rand_vec,
            vmap_method="expand_dims",
        )

        return σp, log_prob_correction


@jit(nopython=True)
def _choose(vp, sections, rand_vec, out, w):
    low_range = 0
    for i, s in enumerate(sections):
        n_rand = low_range + int(np.floor(rand_vec[i] * (s - low_range)))
        out[i] = vp[n_rand]
        w[i] = math.log(s - low_range)
        low_range = s


@struct.dataclass
class HamiltonianRuleJax(HamiltonianRuleBase):
    r"""
    Rule proposing moves according to the terms in an operator.

    In this case, the transition matrix is taken to be:

    .. math::

       T( \mathbf{s} \rightarrow \mathbf{s}^\prime) = \frac{1}{\mathcal{N}(\mathbf{s})}\theta(|H_{\mathbf{s},\mathbf{s}^\prime}|),

    This rule only works with operators which are written in jax.
    """

    operator: DiscreteJaxOperator = struct.field(pytree_node=True)
    """The (hermitian) operator giving the transition amplitudes."""

    def __init__(self, operator: DiscreteJaxOperator):
        if not isinstance(operator, DiscreteJaxOperator):
            raise TypeError(
                "Argument to HamiltonianRule must be a valid operator, "
                f"but operator is a {type(operator)}."
            )
        self.operator = operator

    def transition(self, _0, _1, _2, _3, key, x):
        xp, mels = self.operator.get_conn_padded(x)

        n_conn = self.operator.n_conn(x)

        # def _n_conn(mels):
        #     nonzeros = jnp.abs(mels) > 0
        #     return nonzeros.sum(axis=-1)
        # n_conn = _n_conn(mels)

        rand_i = jax.random.randint(key, shape=(x.shape[0],), minval=0, maxval=n_conn)

        # we need to shift rand_i so that we only select the nonzeros
        #
        # if we were to assume that the nonzero mels are gathered at the beginning of each
        # row we could avoid this, i.e. just do
        # x_proposed = xp[jnp.arange(xp.shape[0]), rand_i]
        #
        # instead of actually shifting we build a mask for the selected element in each row
        # and sum over the zeros at the end

        # TODO let the operator supply the nonzeros or nonzero_i_plus1 directly
        nonzeros = jnp.abs(mels) > 0
        nonzero_i_plus1 = (jnp.cumsum(nonzeros, axis=-1)) * nonzeros
        rand_i_mask = nonzero_i_plus1 == jnp.expand_dims(rand_i + 1, -1)
        # .sum promotes the dtype, so we must convert it to xp dtype
        x_proposed = (xp * jnp.expand_dims(rand_i_mask, -1)).sum(axis=1, dtype=xp.dtype)
        n_conn_proposed = self.operator.n_conn(x_proposed)

        # _, mels_proposed = self.operator.get_conn_padded(x_proposed)
        # n_conn_proposed = _n_conn(mels_proposed)

        # if there are no connected elements x_proposed might
        # contain illegal states (i.e. zeros)
        # and log_prob_corr will be nan
        # TODO is it safe to assume that n_conn >=1 ?
        #
        # no_nonzeros = nonzeros.sum(axis=-1) == 0
        # x_proposed = jax.lax.select(
        #     jax.lax.broadcast_in_dim(no_nonzeros, x.shape, (0,)), x, x_proposed
        # )
        # n_conn = jnp.maximum(n_conn, 1)
        # n_conn_proposed = jnp.maximum(n_conn_proposed, 1)

        log_prob_corr = jnp.log(n_conn) - jnp.log(n_conn_proposed)

        return x_proposed.astype(x.dtype), log_prob_corr


def HamiltonianRule(operator):
    r"""
    Rule proposing moves according to the terms in an operator.

    In this case, the transition matrix is taken to be:

    .. math::

       T( \mathbf{s} \rightarrow \mathbf{s}^\prime) =
        \frac{1}{\mathcal{N}(\mathbf{s})}\theta(|H_{\mathbf{s},\mathbf{s}^\prime}|),

    This is a thin wrapper on top of the constructors of :class:`netket.sampler.rules.HamiltonianRuleJax` and
    :class:`netket.sampler.rules.HamiltonianRuleNumba`, which dispatches on one of the two implementations
    depending on whether the operator specified is jax-compatible (:class:`netket.operator.DiscreteJaxOperator`)
    or not.
    """
    if isinstance(operator, DiscreteJaxOperator):
        return HamiltonianRuleJax(operator)
    else:
        return HamiltonianRuleNumba(operator)
