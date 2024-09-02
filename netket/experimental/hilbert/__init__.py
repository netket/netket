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


__all__ = ["SpinOrbitalFermions"]

from netket.hilbert import SpinOrbitalFermions as _deprecated_SpinOrbitalFermions


_deprecations = {
    # May 2024
    "SpinOrbitalFermions": (
        "netket.experimental.hilbert.SpinOrbitalFermions is deprecated: use "
        "netket.hilbert.SpinOrbitalFermions (netket >= 3.12)",
        _deprecated_SpinOrbitalFermions,
    ),
}

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr

from netket.utils import _hide_submodules

_hide_submodules(__name__)
