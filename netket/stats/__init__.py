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

from netket.stats.mc_stats import statistics, Stats


from netket.stats.mpi_stats import (
    subtract_mean as _deprecated_subtract_mean,
    mean as _deprecated_mean,
    sum as _deprecatd_sum,
    var as _deprecated_var,
    total_size as _deprecated_total_size,
)

from netket.utils import _hide_submodules
from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

_deprecations = {
    # March 2026
    "mean": (
        "netket.stats.mean is deprecated: use jnp.mean directly instead.",
        _deprecated_mean,
    ),
    "var": (
        "netket.stats.var is deprecated: use jnp.var directly instead.",
        _deprecated_var,
    ),
    "sum": (
        "netket.stats.sum is deprecated: use jnp.sum directly instead.",
        _deprecatd_sum,
    ),
    "subtract_mean": (
        "netket.stats.subtract_mean is deprecated: use x-jnp.mean(x) directly instead.",
        _deprecated_subtract_mean,
    ),
    "total_size": (
        "netket.stats.total_size is deprecated: use x.size directly instead.",
        _deprecated_total_size,
    ),
}

__getattr__ = _deprecation_getattr(__name__, _deprecations)

_hide_submodules(__name__)
del _deprecation_getattr
