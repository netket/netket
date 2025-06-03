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


import warnings

__all__ = ["Particle", "SpinOrbitalFermions"]

from .particle import Particle


_deprecations = {
    # May 2024
    "SpinOrbitalFermions": (
        "netket.experimental.hilbert.SpinOrbitalFermions is deprecated: use "
        "netket.hilbert.SpinOrbitalFermions (netket >= 3.12)",
        None,
    ),
}

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr


def __getattr__(name):
    if name == "SpinOrbitalFermions":
        from netket.hilbert import SpinOrbitalFermions as _SpinOrbitalFermions

        warnings.warn(
            "netket.experimental.hilbert.SpinOrbitalFermions is deprecated: use "
            "netket.hilbert.SpinOrbitalFermions (netket >= 3.12)",
            DeprecationWarning,
            stacklevel=2,
        )
        return _SpinOrbitalFermions

    return _deprecation_getattr(__name__, _deprecations)(name)


from netket.utils import _hide_submodules

_hide_submodules(__name__)
