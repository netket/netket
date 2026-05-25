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

from netket.operator._abstract_observable import (
    AbstractObservable as AbstractObservable,
)
from netket._src.observable.renyi2 import (
    Renyi2EntanglementEntropy as _Renyi2EntanglementEntropy,
)
from netket._src.observable.variance import (
    VarianceObservable as _VarianceObservable,
)
from netket._src.observable.infidelity import (
    InfidelityOperator as _InfidelityOperator,
)
_deprecations = {
    # May 2026, NetKet 3.22
    "Renyi2EntanglementEntropy": (
        "netket.experimental.observable.Renyi2EntanglementEntropy is now stable: "
        "use it from netket.observable.Renyi2EntanglementEntropy (netket >= 3.22)",
        _Renyi2EntanglementEntropy,
    ),
    "VarianceObservable": (
        "netket.experimental.observable.VarianceObservable is now stable: "
        "use it from netket.observable.VarianceObservable (netket >= 3.22)",
        _VarianceObservable,
    ),
    "InfidelityOperator": (
        "netket.experimental.observable.InfidelityOperator is now stable: "
        "use it from netket.observable.InfidelityOperator (netket >= 3.22)",
        _InfidelityOperator,
    ),
}

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr
from netket.utils import _hide_submodules

__getattr__ = _deprecation_getattr(__name__, _deprecations)
_hide_submodules(__name__)

del _deprecation_getattr
