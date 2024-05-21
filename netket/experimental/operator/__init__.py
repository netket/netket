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


__all__ = ["FermionOperator2nd", "FermionOperator2ndJax"]

from .pyscf import from_pyscf_molecule


from . import fermion

from netket.operator import FermionOperator2nd as _deprecated_FermionOperator2nd
from netket.operator import FermionOperator2ndJax as _deprecated_FermionOperator2ndJax

_deprecations = {
    # June 2024, NetKet 3.13
    "FermionOperator2nd": (
        "netket.experimental.operator.FermionOperator2nd is deprecated: use "
        "netket.operator.FermionOperator2nd (netket >= 3.13)",
        _deprecated_FermionOperator2nd,
    ),
    "FermionOperator2ndJax": (
        "netket.experimental.operator.FermionOperator2ndJax is deprecated: use "
        "netket.operator.FermionOperator2ndJax (netket >= 3.13)",
        _deprecated_FermionOperator2ndJax,
    ),
}


from netket.utils import _auto_export
from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

__getattr__ = _deprecation_getattr(__name__, _deprecations)
_auto_export(__name__)

del _deprecation_getattr, _auto_export
