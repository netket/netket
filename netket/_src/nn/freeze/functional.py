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
Functional approach: wraps an apply_fun to support parameter freezing.

Analogous to apply_operator/functional.py — the frozen params are stored in a
separate ``"frozen_params"`` key in the variables dict (treated as model_state),
and the wrapped apply_fun merges them back before delegating to the original fun.
"""

from typing import Any, Callable

import flax

from netket import jax as nkjax
from netket._src.nn.freeze.common import (
    freeze_variables,
    merge_params,
    unfreeze_variables,
)


def make_frozen_afun(
    apply_fun: Callable,
    variables: dict,
    is_frozen: Callable[[tuple[str, ...], Any], bool],
) -> tuple[Callable, dict]:
    """
    Wrap an apply function so that a subset of its parameters are frozen.

    Frozen parameters are moved from ``variables["params"]`` to
    ``variables["frozen_params"]``.  Because ``"frozen_params"`` sits outside
    the ``"params"`` collection, NetKet never differentiates through it and the
    optimizer never updates it.  The wrapped function merges them back
    transparently before every forward call.

    This is the functional analogue of :class:`~netket.nn.freeze.FreezeLinenWrapper`
    and mirrors the design of
    :func:`~netket.nn.apply_operator.make_logpsi_op_afun`.

    Args:
        apply_fun: The original ``(variables, x, ...) -> y`` function.
        variables: The current variables dict (typically ``vstate.variables``).
        is_frozen: Callable ``(path, leaf) -> bool``.

    Returns:
        ``(new_apply_fun, new_variables)`` where ``new_variables["params"]``
        only contains trainable parameters and ``new_variables["frozen_params"]``
        contains the frozen ones.
    """
    if (
        isinstance(apply_fun, nkjax.HashablePartial)
        and apply_fun.func is _frozen_wrapper_fun
    ):
        raise NotImplementedError(
            "Freezing a functional variational state more than once is not supported. "
            "Unfreeze it first, then freeze the desired parameter subset in one call."
        )

    new_variables = freeze_variables(variables, is_frozen)
    wrapped_fun = nkjax.HashablePartial(_frozen_wrapper_fun, apply_fun)
    return wrapped_fun, new_variables


def _frozen_wrapper_fun(apply_fun, variables, x, *args, **kwargs):
    """Merge frozen params back into ``"params"`` before calling *apply_fun*."""
    variables_no_frozen, frozen_params = flax.core.pop(variables, "frozen_params")
    merged_params = merge_params(
        dict(variables_no_frozen.get("params", {})), dict(frozen_params)
    )
    full_variables = flax.core.copy(variables_no_frozen, {"params": merged_params})
    return apply_fun(full_variables, x, *args, **kwargs)


def make_unfrozen_afun(
    apply_fun: Callable,
    variables: dict,
) -> tuple[Callable, dict]:
    """
    Restore all frozen parameters to the trainable ``"params"`` collection.

    Inverse of :func:`make_frozen_afun`.  The wrapped apply function is kept
    as-is (it still contains the ``_frozen_wrapper_fun`` logic), but
    ``"frozen_params"`` is set to an empty dict so the wrapper performs a
    no-op merge on every forward call.

    Args:
        apply_fun: The apply function from a frozen variational state.
        variables: The current variables dict (with ``"frozen_params"`` key).

    Returns:
        ``(apply_fun, restored_variables)`` where ``"params"`` contains all
        parameters and ``"frozen_params"`` is empty.
    """
    # unfreeze_variables keeps "frozen_params": {} so that _frozen_wrapper_fun
    # (which calls flax.core.pop(variables, "frozen_params")) still works.
    restored_variables = unfreeze_variables(variables)
    return apply_fun, restored_variables
