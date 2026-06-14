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
    Move the parameters selected by *is_frozen* from ``"params"`` to 
    ``"frozen_params"`` collection, which is not considered a diffable
    parameter.

    Operates on a *flat* variables dict (no wrapper-specific nesting): the
    ``"params"`` and ``"frozen_params"`` collections mirror the natural
    parameter tree of the underlying model.  Any parameters already in
    ``"frozen_params"`` are preserved, so repeated calls accumulate.

    Args:
        variables: A variables dict with at least a ``"params"`` key.
        is_frozen: Callable ``(path, leaf) -> bool``.

    Returns:
        Updated variables dict.

    Example:
        Freeze every ``kernel`` leaf, moving it out of the trainable
        ``"params"`` into ``"frozen_params"``::

            import jax.numpy as jnp
            from netket._src.nn.freeze.common import freeze_variables

            variables = {
                "params": {
                    "Dense_0": {"kernel": jnp.ones((2, 3)), "bias": jnp.zeros(3)},
                }
            }

            # is_frozen receives the path tuple, e.g. ("Dense_0", "kernel")
            frozen = freeze_variables(variables, lambda path, leaf: path[-1] == "kernel")

            # frozen["params"]        == {"Dense_0": {"bias": ...}}      (trainable)
            # frozen["frozen_params"] == {"Dense_0": {"kernel": ...}}    (held fixed)

        Calls accumulate, so a second freeze adds to ``"frozen_params"`` rather
        than replacing it; :func:`unfreeze_variables` reverses all of it.
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
