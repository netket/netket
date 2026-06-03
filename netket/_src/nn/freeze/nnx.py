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

from typing import Callable

from flax import nnx


class Frozen(nnx.Variable):
    """
    NNX Variable subtype that marks parameters as non-trainable.
    """


def _convert_variable_type(
    module: nnx.Module,
    from_type: type,
    to_type: type,
    filter_fn: Callable | None,
) -> nnx.Module:
    """
    Return a *new* module with matching variables retyped ``from_type`` → ``to_type``.
    """
    graphdef, state = nnx.split(module)

    def convert(path, variable):
        if isinstance(variable, from_type) and (
            filter_fn is None or filter_fn(path, variable)
        ):
            return to_type(variable.get_value(), **variable.get_metadata())
        return variable

    return nnx.merge(graphdef, nnx.map_state(convert, state))


def freeze_params(
    module: nnx.Module,
    filter_fn: Callable[[tuple[str, ...], nnx.Param], bool] | None = None,
) -> nnx.Module:
    """
    Return a copy of *module* with ``nnx.Param`` variables converted to :class:`Frozen`.

    Args:
        module: The NNX module to freeze (left unmodified).
        filter_fn: Optional callable ``(path, variable) -> bool`` where *path*
            is a tuple of attribute-name strings and *variable* is the
            ``nnx.Param`` instance.  Only variables for which this returns
            ``True`` are frozen.  ``None`` freezes all ``nnx.Param`` variables.

    Returns:
        A new NNX module with the selected parameters frozen.
    """
    return _convert_variable_type(module, nnx.Param, Frozen, filter_fn)


def unfreeze_params(
    module: nnx.Module,
    filter_fn: Callable[[tuple[str, ...], "Frozen"], bool] | None = None,
) -> nnx.Module:
    """
    Return a copy of *module* with :class:`Frozen` variables converted back to ``nnx.Param``.

    This is the inverse of :func:`freeze_params`.
    Args:
        module: The NNX module to unfreeze (left unmodified).
        filter_fn: Optional callable ``(path, variable) -> bool``.  Only
            variables for which this returns ``True`` are unfrozen.  ``None``
            unfreezes all :class:`Frozen` variables.

    Returns:
        A new NNX module with the selected parameters unfrozen.
    """
    return _convert_variable_type(module, Frozen, nnx.Param, filter_fn)
