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
from collections.abc import Callable, Iterable

from flax.core import FrozenDict


def _append_path(root: Any, key: Any) -> Any:
    return f"{root}/{key}"


def _default_children(tree: Any) -> Iterable[tuple[Any, Any]] | None:
    if isinstance(tree, list):
        return enumerate(tree)
    elif isinstance(tree, tuple) and hasattr(tree, "_fields"):
        return ((key, getattr(tree, key)) for key in tree._fields)
    elif isinstance(tree, tuple):
        return enumerate(tree)
    elif isinstance(tree, dict | FrozenDict):
        return tree.items()
    elif hasattr(tree, "to_compound"):
        return _default_children(tree.to_compound()[1])
    elif hasattr(tree, "to_dict"):
        return _default_children(tree.to_dict())
    else:
        return None


def walk_tree_with_path(
    tree: Any,
    root: Any,
    *,
    visit_leaf: Callable[..., Any],
    enter_node: Callable[..., dict[str, Any] | None] | None = None,
    expand_node: Callable[..., Iterable[tuple[Any, Any]] | None] | None = None,
    path_fn: Callable[[Any, Any], Any] = _append_path,
    **kwargs,
) -> Any:
    """
    Traverse a nested tree while threading a path to each leaf.

    Args:
        tree: the tree to traverse.
        root: the current path accumulator.
        visit_leaf: function applied to leaves as
            ``visit_leaf(root, leaf, **kwargs)``.
        enter_node: optional hook run before descending into a node. It can
            return a dict of keyword arguments to override for that subtree.
            The expected signature is
            ``enter_node(root, node, **kwargs) -> dict[str, Any] | None``.
        expand_node: optional hook returning custom children for a node as
            ``(key, child)`` pairs. Return ``None`` to use the default traversal.
            The expected signature is
            ``expand_node(root, node, **kwargs) -> Iterable[tuple[key, child]] | None``.
        path_fn: function used to build child paths from ``(root, key)``.
        **kwargs: additional keyword arguments forwarded to hooks and leaves.

    Returns:
        The original ``root`` after the traversal completes.

    Example:
        Collect all leaves into a list of ``(path, value)`` pairs:

        .. code-block:: python

            from netket.utils.tree_walk import walk_tree_with_path

            def visit_leaf(path, leaf, *, out, prefix=""):
                full_path = f"{prefix}/{path}" if prefix else path
                out.append((full_path, leaf))

            def enter_node(path, node, *, prefix=""):
                if path == "":
                    return {"prefix": "data"}
                return None

            def path_fn(path, key):
                return f"{path}/{key}" if path else str(key)

            out = []
            walk_tree_with_path(
                {"energy": [1.0, 2.0], "stats": {"mean": 0.5}},
                "",
                visit_leaf=visit_leaf,
                enter_node=enter_node,
                path_fn=path_fn,
                out=out,
            )

        In practice:

        - `visit_leaf` does the work at each leaf.
        - `enter_node` can inject or override state for a whole subtree.
        - `expand_node` is only needed when a node should expose custom children,
          such as splitting a complex number into `("re", value.real)` and
          `("im", value.imag)`.
    """
    if tree is None:
        return root

    if enter_node is not None:
        state_update = enter_node(root, tree, **kwargs)
        if state_update is not None:
            kwargs = kwargs | state_update

    children = None if expand_node is None else expand_node(root, tree, **kwargs)
    if children is None:
        children = _default_children(tree)

    if children is None:
        return visit_leaf(root, tree, **kwargs)

    for key, child in children:
        walk_tree_with_path(
            child,
            path_fn(root, key),
            visit_leaf=visit_leaf,
            enter_node=enter_node,
            expand_node=expand_node,
            path_fn=path_fn,
            **kwargs,
        )

    return root
