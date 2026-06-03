# Copyright 2025 The NetKet Authors - All rights reserved.
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
import jax.numpy as jnp
import pytest
from flax import nnx

import flax.linen as linen_nn

import netket as nk
from netket.vqs import (
    freeze_parameters as freeze_parameters_vstate,
    unfreeze_parameters as unfreeze_parameters_vstate,
)

# Internal symbols — tested but not part of the public API
from netket._src.nn.freeze.nnx import Frozen, freeze_params, unfreeze_params
from netket._src.nn.freeze.linen import FreezeLinenWrapper
from netket._src.nn.freeze.common import (
    freeze_variables as freeze_linen_params,
    unfreeze_variables as unfreeze_linen_params,
)
from netket.utils.model_frameworks.nnx import NNXFramework

# Alias used by the integration tests below
freeze_parameters = freeze_parameters_vstate
unfreeze_parameters = unfreeze_parameters_vstate


class _SimpleModel(nnx.Module):
    def __init__(self):
        self.dense = nnx.Linear(4, 4, rngs=nnx.Rngs(0))
        self.head = nnx.Linear(4, 1, rngs=nnx.Rngs(1))

    def __call__(self, x):
        return self.head(jnp.tanh(self.dense(x)))


# ---------------------------------------------------------------------------
# NNX freeze / unfreeze
# ---------------------------------------------------------------------------


def test_freeze_params_all():
    model = _SimpleModel()
    graphdef, params_before, _ = nnx.split(model, nnx.Param, ...)
    assert len(nnx.to_pure_dict(params_before)) > 0

    model = freeze_params(model)

    graphdef, params_after, state_after = nnx.split(model, nnx.Param, ...)
    assert nnx.to_pure_dict(params_after) == {}
    state_dict = nnx.to_pure_dict(state_after)
    assert "dense" in state_dict
    assert "head" in state_dict


def test_freeze_params_with_filter():
    model = _SimpleModel()

    # Freeze only variables whose path contains "dense"
    model = freeze_params(model, lambda path, _: "dense" in path)

    _, params, state = nnx.split(model, nnx.Param, ...)
    params_dict = nnx.to_pure_dict(params)
    state_dict = nnx.to_pure_dict(state)

    # "head" params are still trainable
    assert "head" in params_dict
    # "dense" params moved to state
    assert "dense" in state_dict
    assert "dense" not in params_dict


def test_unfreeze_params():
    model = _SimpleModel()
    model = freeze_params(model)

    _, params_frozen, _ = nnx.split(model, nnx.Param, ...)
    assert nnx.to_pure_dict(params_frozen) == {}

    model = unfreeze_params(model)

    _, params_restored, state_restored = nnx.split(model, nnx.Param, ...)
    assert nnx.to_pure_dict(state_restored) == {}
    assert "dense" in nnx.to_pure_dict(params_restored)
    assert "head" in nnx.to_pure_dict(params_restored)


def test_frozen_variable_type():
    model = _SimpleModel()
    model = freeze_params(model)
    # After freezing, all params should be Frozen instances
    for _path, node in nnx.graph.iter_graph(model):
        if isinstance(node, nnx.Variable):
            assert isinstance(node, Frozen), f"Expected Frozen, got {type(node)}"


def test_freeze_integrates_with_nnxframework():
    """Frozen params land in model_state, not in the trainable params tree."""
    model = _SimpleModel()

    model = freeze_params(model, lambda path, _: "dense" in path)

    variables, _ = NNXFramework.wrap(model)
    assert "dense" not in variables["params"]
    assert "dense" in variables["model_state"]
    assert "head" in variables["params"]


# ---------------------------------------------------------------------------
# Flax Linen FreezeLinenWrapper
# ---------------------------------------------------------------------------


class _SimpleLinenModel(linen_nn.Module):
    @linen_nn.compact
    def __call__(self, x):
        x = linen_nn.Dense(4)(x)
        return linen_nn.Dense(1)(x).squeeze(-1)


def _init_linen_wrapper(model=None):
    if model is None:
        model = _SimpleLinenModel()
    wrapper = FreezeLinenWrapper(model=model)
    key = jax.random.PRNGKey(0)
    x = jnp.ones((2, 3))
    variables = wrapper.init(key, x)
    return wrapper, variables, x


def test_linen_wrapper_init_structure():
    _, variables, _ = _init_linen_wrapper()
    # share_scope keeps the inner model's params flat at the top level
    assert "model" not in variables["params"]
    assert "Dense_0" in variables["params"]


def test_linen_wrapper_apply_unfrozen():
    wrapper, variables, x = _init_linen_wrapper()
    out = wrapper.apply(variables, x)
    assert out.shape == (2,)


def test_freeze_linen_params_splits_correctly():
    wrapper, variables, x = _init_linen_wrapper()

    frozen_vars = freeze_linen_params(variables, lambda path, _: "kernel" in path)

    # Kernels should be in frozen_params, biases still in params
    model_params = frozen_vars["params"]
    frozen_model = frozen_vars["frozen_params"]

    assert "Dense_0" in model_params  # bias still there
    assert "kernel" not in model_params["Dense_0"]
    assert "bias" in model_params["Dense_0"]
    assert "kernel" in frozen_model["Dense_0"]


def test_freeze_linen_params_preserves_output():
    wrapper, variables, x = _init_linen_wrapper()
    out_before = wrapper.apply(variables, x)

    frozen_vars = freeze_linen_params(variables, lambda path, _: "kernel" in path)
    out_after = wrapper.apply(frozen_vars, x)

    assert jnp.allclose(out_before, out_after)


def test_unfreeze_linen_params_restores():
    wrapper, variables, x = _init_linen_wrapper()
    frozen_vars = freeze_linen_params(variables, lambda path, _: "kernel" in path)
    restored_vars = unfreeze_linen_params(frozen_vars)

    # All params should be back in "params"
    assert "kernel" in restored_vars["params"]["Dense_0"]
    assert restored_vars["frozen_params"] == {}

    # Output should be unchanged
    out_orig = wrapper.apply(variables, x)
    out_restored = wrapper.apply(restored_vars, x)
    assert jnp.allclose(out_orig, out_restored)


def test_freeze_linen_gradients_only_through_trainable():
    wrapper, variables, x = _init_linen_wrapper()
    frozen_vars = freeze_linen_params(variables, lambda path, _: "kernel" in path)

    frozen_params = frozen_vars["frozen_params"]

    def loss(params):
        vars_with_frozen = {
            "params": params,
            "frozen_params": frozen_params,
        }
        return jnp.sum(wrapper.apply(vars_with_frozen, x) ** 2)

    grads = jax.grad(loss)(frozen_vars["params"])
    # Bias has gradient, kernel is not in params (no key)
    assert "bias" in grads["Dense_0"]
    assert "kernel" not in grads["Dense_0"]


class _PositionalArgLinenModel(linen_nn.Module):
    @linen_nn.compact
    def __call__(self, x, scale):
        return linen_nn.Dense(1)(x).squeeze(-1) * scale


def test_linen_wrapper_forwards_positional_args():
    model = _PositionalArgLinenModel()
    key = jax.random.PRNGKey(0)
    x = jnp.ones((2, 3))
    variables = model.init(key, x, 2.0)
    wrapper, wrapped_vars = FreezeLinenWrapper.from_module_and_variables(
        model, variables, lambda path, _: "kernel" in path
    )

    out_before = model.apply(variables, x, 2.0)
    out_after = wrapper.apply(wrapped_vars, x, 2.0)

    assert jnp.allclose(out_before, out_after)


class _MutableLinenModel(linen_nn.Module):
    @linen_nn.compact
    def __call__(self, x):
        counter = self.variable("batch_stats", "counter", lambda: jnp.array(0.0))
        counter.value = counter.value + 1.0
        return linen_nn.Dense(1)(x).squeeze(-1)


def test_linen_wrapper_propagates_mutable_collections():
    model = _MutableLinenModel()
    key = jax.random.PRNGKey(0)
    x = jnp.ones((2, 3))
    variables = model.init(key, x)
    wrapper, wrapped_vars = FreezeLinenWrapper.from_module_and_variables(
        model, variables, lambda path, _: "kernel" in path
    )

    _, updated = wrapper.apply(wrapped_vars, x, mutable=["batch_stats"])

    assert jnp.allclose(
        updated["batch_stats"]["counter"],
        wrapped_vars["batch_stats"]["counter"] + 1.0,
    )


# ---------------------------------------------------------------------------
# freeze_parameters — high-level framework-agnostic API
# ---------------------------------------------------------------------------

_HI = nk.hilbert.Spin(0.5, 4)
_SAMPLER = nk.sampler.MetropolisLocal(_HI, n_chains=4)


class _RBMNNX(nnx.Module):
    def __init__(self):
        self.dense = nnx.Linear(4, 8, rngs=nnx.Rngs(0), use_bias=True)

    def __call__(self, x):
        x = x.astype(jnp.float32)
        return jnp.sum(jnp.log(jnp.cosh(self.dense(x))))


class _RBMLinen(linen_nn.Module):
    @linen_nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        return jnp.sum(jnp.log(jnp.cosh(linen_nn.Dense(8)(x))))


def _make_nnx_vstate():
    return nk.vqs.MCState(_SAMPLER, _RBMNNX(), n_samples=100)


def _make_linen_vstate():
    return nk.vqs.MCState(_SAMPLER, _RBMLinen(), n_samples=100)


def _make_functional_vstate():
    def apply(variables, x):
        x = x.astype(jnp.float32)
        p = variables["params"]
        return jnp.sum(jnp.log(jnp.cosh(x @ p["kernel"] + p["bias"])))

    variables = {
        "params": {
            "kernel": jax.random.normal(jax.random.PRNGKey(0), (4, 8)),
            "bias": jnp.zeros(8),
        }
    }
    return nk.vqs.MCState(_SAMPLER, apply_fun=apply, variables=variables, n_samples=100)


# ---- NNX via freeze_parameters --------------------------------------------


def test_freeze_parameters_nnx_removes_frozen_from_params():
    vstate = _make_nnx_vstate()
    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    assert "kernel" not in frozen.parameters.get("dense", {})
    assert "kernel" in frozen.model_state.get("model_state", {}).get("dense", {})


def test_freeze_parameters_nnx_preserves_output():
    vstate = _make_nnx_vstate()
    x = _HI.all_states()[:2]
    out_before = vstate.log_value(x)
    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    out_after = frozen.log_value(x)
    assert jnp.allclose(out_before, out_after)


def test_unfreeze_parameters_nnx_restores():
    vstate = _make_nnx_vstate()
    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    restored = unfreeze_parameters(frozen)
    assert "kernel" in restored.parameters.get("dense", {})
    # NNX always has a "model_state" collection key even when empty
    assert restored.model_state.get("model_state", {}) == {}


def test_freeze_parameters_preserves_mcstate_runtime_state():
    vstate = _make_nnx_vstate()
    vstate.mutable = ["batch_stats"]
    vstate.training_kwargs = {"deterministic": False}
    vstate._samples = _HI.all_states()[:4]

    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)

    assert frozen.mutable == vstate.mutable
    assert frozen.training_kwargs == vstate.training_kwargs
    assert frozen.sampler_state is vstate.sampler_state
    assert frozen._sampler_state_previous is vstate._sampler_state_previous
    assert frozen._samples is vstate._samples


# ---- Linen via freeze_parameters ------------------------------------------


def test_freeze_parameters_linen_wraps_model():
    vstate = _make_linen_vstate()
    assert not isinstance(vstate._model, FreezeLinenWrapper)
    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    assert isinstance(frozen._model, FreezeLinenWrapper)


def test_freeze_parameters_linen_removes_frozen_from_params():
    vstate = _make_linen_vstate()
    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    assert "kernel" not in frozen.parameters.get("Dense_0", {})
    assert "kernel" in frozen.model_state.get("frozen_params", {}).get("Dense_0", {})


def test_freeze_parameters_linen_preserves_output():
    vstate = _make_linen_vstate()
    x = _HI.all_states()[:2]
    out_before = vstate.log_value(x)
    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    out_after = frozen.log_value(x)
    assert jnp.allclose(out_before, out_after)


def test_freeze_parameters_linen_no_double_wrap():
    """Calling freeze_parameters twice accumulates frozen params without nesting wrappers."""
    vstate = _make_linen_vstate()
    frozen1 = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    frozen2 = freeze_parameters(frozen1, lambda path, _: "bias" in path)
    assert isinstance(frozen2._model, FreezeLinenWrapper)
    # Only one FreezeLinenWrapper layer
    assert not isinstance(frozen2._model.model, FreezeLinenWrapper)
    # Both kernel and bias should now be frozen
    frozen_model = frozen2.model_state.get("frozen_params", {})
    assert "kernel" in frozen_model.get("Dense_0", {})
    assert "bias" in frozen_model.get("Dense_0", {})


def test_freeze_parameters_linen_bound_module():
    """Bound Linen modules should be unbound by MCState then frozen correctly."""
    model = _RBMLinen()
    x = _HI.all_states()[:1].astype(jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), x)
    bound_model = model.bind(variables)
    vstate = nk.vqs.MCState(_SAMPLER, bound_model, n_samples=100)
    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    assert isinstance(frozen._model, FreezeLinenWrapper)
    assert "kernel" not in frozen.parameters.get("Dense_0", {})


def test_freeze_parameters_linen_bound_module_roundtrip():
    """Freeze/unfreeze of a vstate built from a *bound* Linen module is a no-op
    on the output and fully restores the parameters."""
    model = _RBMLinen()
    x_all = _HI.all_states()
    variables = model.init(jax.random.PRNGKey(0), x_all[:1].astype(jnp.float32))
    bound_model = model.bind(variables)
    vstate = nk.vqs.MCState(_SAMPLER, bound_model, n_samples=100)

    out_before = vstate.log_value(x_all[:2])

    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    # Output is unchanged by freezing ...
    assert jnp.allclose(frozen.log_value(x_all[:2]), out_before)
    # ... and the kernel is now non-trainable model state.
    assert "kernel" not in frozen.parameters.get("Dense_0", {})
    assert "kernel" in frozen.model_state.get("frozen_params", {}).get("Dense_0", {})

    restored = unfreeze_parameters(frozen)
    # Unfreezing restores the bare module and every parameter, output preserved.
    assert not isinstance(restored._model, FreezeLinenWrapper)
    assert "kernel" in restored.parameters.get("Dense_0", {})
    assert jnp.allclose(restored.log_value(x_all[:2]), out_before)


def test_unfreeze_parameters_linen_restores():
    vstate = _make_linen_vstate()
    x = _HI.all_states()[:2]
    out_before = vstate.log_value(x)

    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    restored = unfreeze_parameters(frozen)

    # Unfreezing fully unwraps back to the original bare module ...
    assert not isinstance(restored._model, FreezeLinenWrapper)
    # ... with every parameter back in "params" and no frozen collection left.
    assert "kernel" in restored.parameters.get("Dense_0", {})
    assert "frozen_params" not in restored.model_state
    assert jnp.allclose(restored.log_value(x), out_before)


# ---- Functional via freeze_parameters -------------------------------------


def test_freeze_parameters_functional_removes_frozen_from_params():
    vstate = _make_functional_vstate()
    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    assert "kernel" not in frozen.parameters
    assert "kernel" in frozen.model_state.get("frozen_params", {})


def test_freeze_parameters_functional_preserves_output():
    vstate = _make_functional_vstate()
    x = _HI.all_states()[:2]
    out_before = vstate.log_value(x)
    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    out_after = frozen.log_value(x)
    assert jnp.allclose(out_before, out_after)


def test_unfreeze_parameters_functional_restores():
    vstate = _make_functional_vstate()
    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)
    restored = unfreeze_parameters(frozen)
    assert "kernel" in restored.parameters
    # frozen_params is kept as {} so the wrapper function still works
    assert restored.model_state.get("frozen_params", None) == {}


def test_freeze_parameters_functional_twice_not_implemented():
    vstate = _make_functional_vstate()
    frozen = freeze_parameters(vstate, lambda path, _: "kernel" in path)

    with pytest.raises(NotImplementedError, match="more than once"):
        freeze_parameters(frozen, lambda path, _: "bias" in path)
