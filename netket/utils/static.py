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

from netket.utils import struct


@struct.dataclass(cache_hash=True)
class Static:
    """
    This class wraps an hashable class to make it a Static Argument for a
    jax compiled function.
    """

    value: Any = struct.field(pytree_node=False)
    """The wrapped object."""
