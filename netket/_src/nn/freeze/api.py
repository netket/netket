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

from typing import Any, Callable

from flax import linen, nnx

from netket._src.nn.freeze.common import freeze_variables, unfreeze_variables
from netket._src.nn.freeze.nnx import (
    freeze_params as nnx_freeze_params,
    unfreeze_params as nnx_unfreeze_params,
)
from netket._src.nn.freeze.linen import FreezeLinenWrapper
from netket._src.nn.freeze.functional import make_frozen_afun, make_unfrozen_afun


def freeze_parameters(
    model: Any,
    variables: dict,
    is_frozen: Callable[[tuple[str, ...], Any], bool],
) -> tuple[Any, dict]:
    """
    Freeze a subset of model parameters, identified by a filter function.

    This function supports {class}`flax.linen.Module`, {class}`flax.nnx.Module`,
    and simple functions in the following way:

    - **NNX modules**: matching ``nnx.Param`` variables are converted to
      :class:`~netket._src.nn.freeze.nnx.Frozen`, which NetKet puts in the
      model_state dictionary instead of the parameters.
    - **Flax Linen modules**: frozen parameters land in the
      ``"frozen_params"`` collection.
    - **Plain apply functions** (any callable that is not a Module): frozen
      parameters live in ``"frozen_params"``.

    Args:
        model: A module, or a plain ``(variables, x) -> y`` apply function.
        variables: Variables dict matching *model* (with at least a
            ``"params"`` key for Linen and functional models; ignored for NNX,
            whose parameters are stored inside the module).
        is_frozen: Callable filter ``(path, leaf) -> bool``.

    Returns:
        ``(new_model, new_variables)`` — *new_model* has the same calling
        convention as *model* (Module → Module, apply_fun → apply_fun).
        ``new_variables`` is ``None`` for NNX modules (parameters live inside
        the module).

    For variational states, prefer calling :func:`netket.vqs.freeze_parameters`,
    which operates on the variational state itself.

    Example (NNX)::

        import jax.numpy as jnp
        import netket as nk
        from flax import nnx

        class RBM(nnx.Module):
            def __init__(self):
                self.dense = nnx.Linear(4, 8, rngs=nnx.Rngs(0))

            def __call__(self, x):
                return jnp.sum(jnp.log(jnp.cosh(self.dense(x.astype(float)))))

        new_model, new_variables = nk.nn.freeze_parameters(
            RBM(), {}, lambda path, _: "kernel" in path
        )
        # new_variables is None for NNX — params travel inside new_model
        vstate = nk.vqs.MCState(sampler, new_model, variables=new_variables)

    """
    if isinstance(model, nnx.Module):
        return nnx_freeze_params(model, is_frozen), None

    if isinstance(model, linen.Module):
        if isinstance(model, FreezeLinenWrapper):
            # Already wrapped — only split params, accumulating with any
            # previously frozen subset.
            return model, freeze_variables(variables, is_frozen)
        return FreezeLinenWrapper.from_module_and_variables(model, variables, is_frozen)

    if callable(model):
        new_apply_fun, new_variables = make_frozen_afun(model, variables, is_frozen)
        return new_apply_fun, new_variables

    raise TypeError(
        f"freeze_parameters: unsupported model type {type(model)!r}. "
        "Expected a Flax Linen module, an NNX module, or a callable apply_fun."
    )


def unfreeze_parameters(model: Any, variables: dict) -> tuple[Any, dict]:
    """
    Restore all frozen parameters to the trainable ``"params"`` collection.

    Inverse of :func:`freeze_parameters`.

    Args:
        model: A model previously returned by :func:`freeze_parameters`.
        variables: The corresponding variables dict.

    Returns:
        ``(new_model, new_variables)`` with every parameter trainable again.
    """
    if isinstance(model, nnx.Module):
        return nnx_unfreeze_params(model), None

    if isinstance(model, linen.Module):
        if isinstance(model, FreezeLinenWrapper):
            return model.unfreeze(variables)
        # Plain (never-frozen) Linen module: nothing to unwrap, just merge.
        return model, unfreeze_variables(variables)

    if callable(model):
        new_apply_fun, new_variables = make_unfrozen_afun(model, variables)
        return new_apply_fun, new_variables

    raise TypeError(f"unfreeze_parameters: unsupported model type {type(model)!r}.")
