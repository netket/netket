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
Internal utility functions to support jax sharding natively within netket.
All functions in here are not part of the public API, internal, and may change without warning.
"""

from netket.jax.sharding.debug import inspect as inspect

from netket.jax.sharding.fixed_sharding_decorator import (
    sharding_decorator as sharding_decorator,
    _increase_SHARD_MAP_STACK_LEVEL as _increase_SHARD_MAP_STACK_LEVEL,
    _get_SHARD_MAP_STACK_LEVEL as _get_SHARD_MAP_STACK_LEVEL,
)

from netket.jax.sharding.fixed_sharding_utils import (
    shard_along_axis as shard_along_axis,
    with_samples_sharding_constraint as with_samples_sharding_constraint,
    extract_replicated as extract_replicated,
    distribute_to_devices_along_axis as distribute_to_devices_along_axis,
    gather as gather,
)

from netket.jax.sharding.flexible_sharding import (
    pad_axis_for_sharding as pad_axis_for_sharding,
)


from netket.utils import _hide_submodules
from netket.utils.deprecation import warn_deprecation as _warn_deprecation


# TODO: Deprecated in July 2025, remove eventually. It was internal
def __getattr__(name):
    """Handle deprecated attribute access with warnings."""
    if name == "SHARD_MAP_STACK_LEVEL":
        _warn_deprecation(
            "Accessing netket.jax.sharding.SHARD_MAP_STACK_LEVEL directly is deprecated. Use _get_SHARD_MAP_STACK_LEVEL() instead.",
        )
        return _get_SHARD_MAP_STACK_LEVEL()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


_hide_submodules(__name__)
