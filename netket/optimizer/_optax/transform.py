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
import optax
from jax import numpy as jnp

from . import utils


def _update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order-th` moment."""
    return jax.tree_map(
        lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments
    )


def _update_moment_norm(updates, moments, decay, order):
    """Compute the exponential moving average of the `order-th` moment of the norm."""

    def orderth_norm(g):
        if jnp.isrealobj(g):
            return g ** order
        else:
            return (g.conj() * g).real ** (order / 2)

    return jax.tree_map(
        lambda g, t: (1 - decay) * orderth_norm(g) + decay * t, updates, moments
    )


def _bias_correction(moment, decay, count):
    """Perform bias correction. This becomes a no-op as count goes to infinity."""
    bias_correction = 1 - decay ** count
    return jax.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0,
    mu_dtype=None,
) -> optax.GradientTransformation:
    """Rescale updates according to the Adam algorithm.
    References:
      [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)
    Args:
      b1: decay rate for the exponentially weighted average of grads.
      b2: decay rate for the exponentially weighted average of squared norm of grads.
      eps: term added to the denominator to improve numerical stability.
      eps_root: term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
      mu_dtype: optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype is inferred from `params` and `updates`.
    Returns:
      An (init_fn, update_fn) tuple.
    """
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu = jax.tree_map(jnp.zeros_like, params)
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = _update_moment(updates, state.mu, b1, 1)
        nu = _update_moment_norm(updates, state.nu, b2, 2)
        count_inc = utils.safe_int32_increment(state.count)
        mu_hat = utils.cast_tree(_bias_correction(mu, b1, count_inc), mu_dtype)
        nu_hat = _bias_correction(nu, b2, count_inc)
        updates = jax.tree_map(
            lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat
        )
        return updates, optax.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)
