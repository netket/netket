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


"""
This file only contains deprecated bindings. To be removed.
"""

from netket.operator import fermion as _deprecated_fermion

_deprecations = {
    # June 2024, NetKet 3.13
    f"{_name}": (
        f"netket.experimental.operator.fermion.{_name} is deprecated: use "
        f"netket.operator.fermion.{_name} (netket >= 3.13)",
        getattr(_deprecated_fermion, _name),
    )
    for _name in [
        "destroy",
        "create",
        "number",
        "identity",
        "zero",
    ]
}

from netket.utils import _auto_export  # noqa: E402
from netket.utils.deprecation import (  # noqa: E402
    deprecation_getattr as _deprecation_getattr,
)

__getattr__ = _deprecation_getattr(__name__, _deprecations)
_auto_export(__name__)
