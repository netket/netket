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

"""Internal helpers shared by the Linen and functional freeze implementations."""

from typing import Any, Callable

from flax import traverse_util


def split_params(
    params: Any,
    is_frozen: Callable[[tuple[str, ...], Any], bool],
) -> tuple[dict, dict]:
    """
    Partition a (possibly nested) parameter dict into ``(trainable, frozen)``.

    Args:
        params: A nested dict of parameters.
        is_frozen: Callable ``(path, leaf) -> bool`` where *path* is the tuple
            of string keys leading to *leaf*.  Leaves for which this returns
            ``True`` go into the frozen sub-tree, the rest into the trainable one.

    Returns:
        ``(trainable, frozen)`` — two nested dicts with disjoint leaves whose
        union reconstructs *params*.
    """
    trainable: dict = {}
    frozen: dict = {}
    for path, leaf in traverse_util.flatten_dict(dict(params)).items():
        (frozen if is_frozen(path, leaf) else trainable)[path] = leaf
    return (
        traverse_util.unflatten_dict(trainable),
        traverse_util.unflatten_dict(frozen),
    )


def merge_params(*trees: Any) -> dict:
    """
    Merge parameter dicts into a single nested dict.

    Later trees fill in (or override) leaves from earlier ones.  Used to
    recombine disjoint trainable and frozen sub-trees.
    """
    flat: dict = {}
    for tree in trees:
        flat.update(traverse_util.flatten_dict(dict(tree)))
    return traverse_util.unflatten_dict(flat)


def freeze_variables(
    variables: dict,
    is_frozen: Callable[[tuple[str, ...], Any], bool],
) -> dict:
    """
    Move the parameters selected by *is_frozen* from ``"params"`` to ``"frozen_params"``.

    Operates on a *flat* variables dict (no wrapper-specific nesting): the
    ``"params"`` and ``"frozen_params"`` collections mirror the natural
    parameter tree of the underlying model.  Any parameters already in
    ``"frozen_params"`` are preserved, so repeated calls accumulate.

    Shared by the Linen (:mod:`~netket._src.nn.freeze.linen`) and functional
    (:mod:`~netket._src.nn.freeze.functional`) backends, which both freeze by
    moving leaves between collections.

    Args:
        variables: A variables dict with at least a ``"params"`` key.
        is_frozen: Callable ``(path, leaf) -> bool``.

    Returns:
        Updated variables dict.
    """
    trainable, newly_frozen = split_params(variables.get("params", {}), is_frozen)
    combined_frozen = merge_params(variables.get("frozen_params", {}), newly_frozen)

    new_variables = dict(variables)
    new_variables["params"] = trainable
    new_variables["frozen_params"] = combined_frozen
    return new_variables


def unfreeze_variables(variables: dict) -> dict:
    """
    Restore every leaf in ``"frozen_params"`` to the trainable ``"params"`` collection.

    Inverse of :func:`freeze_variables`.  ``"frozen_params"`` is left as an empty
    dict rather than removed, so wrappers that always expect the key (e.g. the
    functional backend) keep working.
    """
    merged = merge_params(
        variables.get("params", {}),
        variables.get("frozen_params", {}),
    )

    new_variables = dict(variables)
    new_variables["params"] = merged
    new_variables["frozen_params"] = {}
    return new_variables
