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
