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

import optax

from . import transform


def _scale_by_learning_rate(learning_rate, flip_sign=True):
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return optax.scale_by_schedule(lambda count: m * learning_rate(count))
    return optax.scale(m * learning_rate)


def adam(
    learning_rate,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype=None,
) -> optax.GradientTransformation:
    """The classic Adam optimiser.

    Adam is an SGD variant with learning rate adaptation. The `learning_rate`
    used for each weight is computed from estimates of first- and second-order
    moments of the gradients (using suitable exponential moving averages).

    References:
      Kingma et al, 2014: https://arxiv.org/abs/1412.6980

    Args:
      learning_rate: this is a fixed global scaling factor.
      b1: the exponential decay rate to track the first moment of past gradients.
      b2: the exponential decay rate to track the second moment of past gradients.
      eps: a small constant applied to denominator outside of the square root
        (as in the Adam paper) to avoid dividing by zero when rescaling.
      eps_root: (default `0`), a small constant applied to denominator inside the
        square root (as in RMSProp), to avoid dividing by zero when rescaling.
        This is needed for example when computing (meta-)gradients through Adam.
      mu_dtype: optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype is inferred from `params` and `updates`.

    Returns:
      the corresponding `GradientTransformation`.
    """
    return optax.chain(
        transform.scale_by_adam(
            b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype
        ),
        _scale_by_learning_rate(learning_rate),
    )
