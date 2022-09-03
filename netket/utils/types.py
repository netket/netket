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

from typing import Any, Sequence, Callable, Union

import jax as _jax
import jaxlib as _jaxlib
import numpy as _np

PRNGKeyT = Any
SeedT = Union[int, PRNGKeyT]

Shape = Sequence[int]
DType = Any  # this could be a real type?

Array = Union[_np.ndarray, _jaxlib.xla_extension.DeviceArray, _jax.core.Tracer]
ArrayLike = Any  # Objects that are valid inputs to (np|jnp).asarray.

NNInitFunc = Callable[[PRNGKeyT, Shape, DType], Array]

PyTree = Any

Scalar = Any
