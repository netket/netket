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
import flax
import numpy as np

from jax import numpy as jnp
from jax.experimental import host_callback as hcb
from flax import struct
from numba import jit

import math

from typing import Any

from netket.operator import AbstractOperator
from netket.jax import numba_to_jax, njit4jax

from ..metropolis import MetropolisRule


@struct.dataclass
class HamiltonianRule(MetropolisRule):
    """
    Rule proposing moves according to the terms in an operator.

    In this case, the transition matrix is taken to be:

    .. math::

       T( \\mathbf{s} \\rightarrow \\mathbf{s}^\\prime) = \\frac{1}{\\mathcal{N}(\\mathbf{s})}\\theta(|H_{\\mathbf{s},\\mathbf{s}^\\prime}|),

    This rule only works on CPU! If you want to use it on GPU, you
    must use the numpy variant :class:`netket.sampler.rules.HamiltonianRuleNumpy`
    together with the numpy metropolis sampler :class:`netket.sampler.MetropolisSamplerNumpy`.
    """

    operator: AbstractOperator = struct.field(pytree_node=False)
    """The (hermitian) operator giving the transition amplitudes."""

    def init_state(rule, sampler, machine, params, key):
        if sampler.hilbert != rule.operator.hilbert:
            raise ValueError(
                f"""
            The hilbert space of the sampler ({sampler.hilbert}) and the hilbert space
            of the operator ({rule.operator.hilbert}) for HamiltonianRule must be the same.
            """
            )
        return super().init_state(sampler, machine, params, key)

    def __post_init__(self):
        # Raise errors if hilbert is not an Hilbert
        if not isinstance(self.operator, AbstractOperator):
            raise TypeError(
                "Argument to HamiltonianRule must be a valid operator.".format(
                    type(self.operator)
                )
            )

    def transition(rule, sampler, machine, parameters, state, key, σ):

        hilbert = sampler.hilbert
        get_conn_flattened = rule.operator._get_conn_flattened_closure()
        n_conn_from_sections = rule.operator._n_conn_from_sections

        @njit4jax(
            (
                jax.abstract_arrays.ShapedArray(σ.shape, σ.dtype),
                jax.abstract_arrays.ShapedArray((σ.shape[0],), σ.dtype),
            )
        )
        def _transition(args):
            # unpack arguments
            v_proposed, log_prob_corr, v, rand_vec = args

            log_prob_corr.fill(0)
            sections = np.empty(v.shape[0], dtype=np.int32)
            vp, _ = get_conn_flattened(v, sections)

            _choose(vp, sections, rand_vec, v_proposed, log_prob_corr)

            # TODO: n_conn(v_proposed, sections) implemented below, but
            # might be slower than fast implementations like ising
            get_conn_flattened(v_proposed, sections)
            n_conn_from_sections(sections)

            log_prob_corr -= np.log(sections)

        # ideally we would pass the key to python/numba in _choose, initialise a
        # np.random.default_rng(key) and use it to generatee random uniform integers.
        # However, numba dose not support np states, and reseeding it's MT1998 implementation
        # would be slow so we generate floats in the [0,1] range in jax and pass those
        # to python
        rand_vec = jax.random.uniform(key, shape=(σ.shape[0],))

        σp, log_prob_correction = _transition(σ, rand_vec)

        return σp, log_prob_correction

    def __repr__(self):
        return f"HamiltonianRule({self.operator})"


@jit(nopython=True)
def _choose(vp, sections, rand_vec, out, w):
    low_range = 0
    for i, s in enumerate(sections):
        n_rand = low_range + int(np.floor(rand_vec[i] * (s - low_range)))
        out[i] = vp[n_rand]
        w[i] = math.log(s - low_range)
        low_range = s
