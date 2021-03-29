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

from jax import numpy as jnp
from flax.core import FrozenDict


def init_fun(model, size, key, dtype=jnp.float32):
    """
    Default initializer to get the weights of a flax
    model.
    """
    dummy_input = jnp.zeros((1, size), dtype=dtype)

    variables = model.init({"params": key}, dummy_input)
    model_state, params = variables.pop("params")

    return model_state, params


def evaluate_model_fun(model, variables, σ):
    """
    Default function to evaluate the model with
    variables at input σ.
    """
    return model.apply(variables, σ)


def evaluate_mutable_model_fun(model, variables, σ, mutable):
    """
    Default function to evaluate the model during training, with
    mutable argument passed so that mutated state can be passed back
    """
    return model.apply(variables, σ, mutable=mutable)


def apply_with_state_fun(apply_fun, model_state):
    return lambda w, σ: apply_fun({"params": w, **model_state}, σ)
