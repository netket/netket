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

from typing import Any
from collections.abc import Callable
from functools import partial


from flax.core import FrozenDict

from ..dispatch import dispatch

from .history import History

AccumulatorFun = Callable[[Any, Any], Any]

AccumulatorFunTypeRegistry: dict[type, AccumulatorFun] = {}
"""
Dictionary used for dispatching the accumulator functino based on the type
of the input tree.
"""


def _accum_histories_container(
    fun: Callable[[Any, Any], Any],
    tree_accum: list | Any,
    tree: list | Any,
    container_type: type = list,
    **kwargs,
):
    # If nothing to accumulate, just return the former tree
    if len(tree) == 0:
        return tree_accum

    if tree_accum is None:
        tree_accum = [None for _ in range(len(tree))]
    return container_type(
        accum_in_tree(fun, _accum, _tree, **kwargs)
        for _accum, _tree in zip(tree_accum, tree)
    )


def _accum_histories_dict(
    fun: Callable[[Any, Any], Any],
    tree_accum: list | Any,
    tree: list,
    **kwargs,
):
    # If nothing to accumulate, just return the former tree
    if len(tree.keys()) == 0:
        return tree_accum

    if tree_accum is None:
        tree_accum = {}
    for key in tree.keys():
        value = accum_in_tree(fun, tree_accum.get(key, None), tree[key], **kwargs)
        #
        if value is not None:
            tree_accum[key] = value
    return tree_accum


AccumulatorFunTypeRegistry[list] = partial(
    _accum_histories_container, container_type=list
)
AccumulatorFunTypeRegistry[tuple] = partial(
    _accum_histories_container, container_type=tuple
)
AccumulatorFunTypeRegistry[dict] = _accum_histories_dict
AccumulatorFunTypeRegistry[FrozenDict] = _accum_histories_dict


def accum_in_tree(
    fun: Callable[[Any, Any], Any],
    tree_accum: dict[str, Any] | Any,
    tree: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """
    Maps all the leafs in the two trees, applying the function with the leafs of tree1
    as first argument and the leafs of tree2 as second argument
    Any additional argument after the first two is forwarded to the function call.

    This is useful e.g. to sum the leafs of two trees

    Args:
        fun: the function to apply to all leafs
        tree1: the structure containing leafs. This can also be just a leaf
        tree2: the structure containing leafs. This can also be just a leaf
        *args: additional positional arguments passed to fun
        **kwargs: additional kw arguments passed to fun

    Returns:
        An equivalent tree, containing the result of the function call.
    """
    if tree is None:
        return tree_accum

    tree_type = type(tree)
    if tree_type in AccumulatorFunTypeRegistry:
        return AccumulatorFunTypeRegistry[tree_type](fun, tree_accum, tree, **kwargs)
    elif hasattr(tree, "to_compound"):
        return fun(tree_accum, tree, **kwargs)
    elif hasattr(tree, "to_dict"):
        return accum_in_tree(fun, tree_accum, tree.to_dict(), **kwargs)
    else:
        return fun(tree_accum, tree, **kwargs)


@dispatch
def init_history_from_data(val: Any, step: Any):
    return History(val, step)


def accum_histories(accum: History | None, data, *, step=0) -> History:
    if accum is None:
        return init_history_from_data(data, step)
    else:
        accum.append(data, it=step)
        return accum


accum_histories_in_tree = partial(accum_in_tree, accum_histories)
