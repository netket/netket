# Copyright 2021 The NetKet Authors - All rights reserved.

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

from typing import Any, Union, Protocol
from collections.abc import Callable
from collections.abc import Sequence

import optax as _optax
import jax as _jax
import numpy as _np
import flax as _flax

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

PRNGKeyT = Any
SeedT = Union[int, PRNGKeyT]


class _SupportsDType(Protocol):
    @property
    def dtype(self) -> _np.dtype: ...


Shape = Sequence[int]
DType = Union[
    None,
    str,  # like 'float32', 'int32'
    type[Any],  # like np.float32, np.int32, float, int
    _np.dtype,  # like np.dtype('float32'), np.dtype('int32')
    _SupportsDType,  # like jnp.float32, jnp.int32
]

Array = Union[_np.ndarray, _jax.Array]
JaxArray = _jax.Array

ArrayLike = Any  # Objects that are valid inputs to (np|jnp).asarray.

NNInitFunc = Callable[[PRNGKeyT, Shape, DType], _jax.Array]

PyTree = Any
ModuleOrApplyFun = Union[_flax.linen.Module, Callable[[PyTree, Array], Array]]
ModelStateT = dict[str, PyTree]

Scalar = Any

ScalarOrSchedule = Union[Scalar, _optax.Schedule]

Optimizer = Any

_deprecations = {
    # April 2025
    "Any": (
        "netket.utils.types.Any is deprecated: use " "typing.Any (netket >= 3.17)",
        Any,
    ),
    "Union": (
        "netket.utils.types.Union is deprecated: use "
        "typing.Union or `|` (netket >= 3.17)",
        Union,
    ),
    "Protocol": (
        "netket.utils.types.Protocol is deprecated: use "
        "typing.Protocol (netket >= 3.17)",
        Protocol,
    ),
    "Callable": (
        "netket.utils.types.Callable is deprecated: use "
        "collections.abc.Callable (netket >= 3.17)",
        Callable,
    ),
    "Sequence": (
        "netket.utils.types.Sequence is deprecated: use "
        "collections.abc.Sequence (netket >= 3.17)",
        Sequence,
    ),
}

__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr


del Any, Union, Protocol, Callable, Sequence
