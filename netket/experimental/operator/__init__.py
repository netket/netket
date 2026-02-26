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


__all__ = [
    "from_pyscf_molecule",
    "fermion",
    "pyscf",
    "FermiHubbardJax",
    "ParticleNumberConservingFermioperator2nd",
    "ParticleNumberAndSpinConservingFermioperator2nd",
]

from netket._src.operator.pyscf_api import from_pyscf_molecule as from_pyscf_molecule
from netket._src.operator.particle_number_conserving_fermionic.operators import (
    ParticleNumberConservingFermioperator2nd,
    ParticleNumberAndSpinConservingFermioperator2nd,
)

from netket.experimental.operator import pyscf as pyscf

from netket._src.operator.particle_number_conserving_fermionic.fermihubbard import (
    FermiHubbardJax as _deprecated_FermiHubbardJax,
)

_deprecations = {
    # March 2026, NetKet 3.21
    "FermiHubbardJax": (
        "netket.experimental.operator.FermiHubbardJax is now stable: use "
        "netket.operator.FermiHubbardJax",
        _deprecated_FermiHubbardJax,
    ),
}

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr
from netket.utils import _auto_export

__getattr__ = _deprecation_getattr(__name__, _deprecations)
_auto_export(__name__)

del _deprecation_getattr, _auto_export
