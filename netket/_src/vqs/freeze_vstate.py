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

"""
Variational-state-level freeze/unfreeze.

Thin shim over :func:`netket.nn.freeze_parameters` that constructs a new
:class:`~netket.vqs.MCState` or :class:`~netket.vqs.FullSumState`.
"""

from typing import Any, Callable

from netket.vqs import FullSumState, MCState, VariationalState

from netket._src.nn.freeze.api import (
    freeze_parameters as _freeze_model,
    unfreeze_parameters as _unfreeze_model,
)


def _is_module_model(model) -> bool:
    """True if *model* is a Flax/NNX module (i.e. not a plain apply_fun).

    The freeze dispatcher only ever returns a real ``linen.Module`` /
    ``nnx.Module`` or a plain apply function — never NetKet's ``NNXWrapper``.
    """
    from flax import linen, nnx

    return isinstance(model, (linen.Module, nnx.Module))


def _rebuild_vstate(
    vstate: VariationalState,
    new_model_or_fun: Any,
    new_variables: dict | None,
) -> VariationalState:
    """Build a new vstate of the same type as *vstate*, swapping model+variables.

    For NNX models the freeze dispatcher returns ``new_variables=None`` (the
    parameters live inside the module); MCState/FullSumState then re-wrap the
    module and extract the params themselves.
    """
    if _is_module_model(new_model_or_fun):
        new_model = new_model_or_fun
        new_apply_fun = None
    else:
        new_model = None
        new_apply_fun = new_model_or_fun

    if isinstance(vstate, FullSumState):
        return FullSumState(
            hilbert=vstate.hilbert,
            model=new_model,
            apply_fun=new_apply_fun,
            variables=new_variables,
            chunk_size=vstate.chunk_size,
            mutable=vstate.mutable,
            training_kwargs=dict(vstate.training_kwargs),
        )

    if isinstance(vstate, MCState):
        new_vstate = MCState(
            sampler=vstate.sampler,
            model=new_model,
            apply_fun=new_apply_fun,
            variables=new_variables,
            n_samples=vstate.n_samples,
            n_discard_per_chain=vstate.n_discard_per_chain,
            chunk_size=vstate.chunk_size,
            mutable=vstate.mutable,
            training_kwargs=dict(vstate.training_kwargs),
        )
        new_vstate.sampler_state = vstate.sampler_state
        new_vstate._sampler_state_previous = vstate._sampler_state_previous
        if vstate._samples is not None:
            new_vstate._samples = vstate._samples
        return new_vstate

    raise TypeError(
        f"Unsupported variational state type {type(vstate)!r}. "
        "Expected MCState or FullSumState."
    )


def _extract_model_or_fun(vstate: VariationalState) -> Any:
    """
    Return the *real* model (Linen / NNX module) or plain apply function.

    NetKet stores NNX models internally as an
    :class:`~netket.utils.model_frameworks.nnx.NNXWrapper`.  We unwrap it back
    to a genuine ``nnx.Module`` here (via the framework-aware ``vstate.model``
    property) so the framework-agnostic :mod:`netket._src.nn.freeze` machinery
    never has to know about NetKet's wrapper types.

    Linen models are returned as the *unbound* static module (``vstate._model``)
    rather than ``vstate.model``, which would re-bind a bound module.
    """
    from netket.utils.jax import WrappedApplyFun
    from netket.utils.model_frameworks.nnx import NNXWrapper

    model = vstate._model
    if model is None:
        return vstate._apply_fun
    if isinstance(model, WrappedApplyFun):
        return model.apply
    if isinstance(model, NNXWrapper):
        return vstate.model
    return model


def freeze_parameters(
    vstate: VariationalState,
    is_frozen: Callable[[tuple[str, ...], Any], bool],
) -> VariationalState:
    """
    Freeze a subset of model parameters in a variational state.

    Thin shim around :func:`netket.nn.freeze_parameters`: extracts the
    model and variables from *vstate*, freezes the matching parameters, and
    returns a new variational state of the same type.  Frozen parameters are
    moved from ``vstate.parameters`` into ``vstate.model_state`` so they are
    automatically excluded from gradient computation and optimizer updates.

    Args:
        vstate: A :class:`~netket.vqs.MCState` or
            :class:`~netket.vqs.FullSumState`.
        is_frozen: Callable ``(path, leaf) -> bool``.

    Returns:
        A new variational state of the same type as *vstate*.

    Example::

        import netket as nk

        frozen_vstate = nk.vqs.freeze_parameters(
            vstate, lambda path, _: "kernel" in path
        )
    """
    new_model_or_fun, new_variables = _freeze_model(
        _extract_model_or_fun(vstate), vstate.variables, is_frozen
    )
    return _rebuild_vstate(vstate, new_model_or_fun, new_variables)


def unfreeze_parameters(vstate: VariationalState) -> VariationalState:
    """
    Restore all frozen parameters in *vstate* to the trainable set.

    Inverse of :func:`freeze_parameters`.
    """
    new_model_or_fun, new_variables = _unfreeze_model(
        _extract_model_or_fun(vstate), vstate.variables
    )
    return _rebuild_vstate(vstate, new_model_or_fun, new_variables)
