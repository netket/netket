# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from netket.utils import wraps_legacy
from netket.legacy.optimizer import SR as SR_legacy
from netket.legacy.machine import AbstractMachine

from .sr_onthefly import SR_otf_cg, SR_otf_gmres

default_iterative = "cg"
# default_direct = "eigen"


@wraps_legacy(SR_legacy, "machine", AbstractMachine)
def SR(diag_shift=0.01, method=None, *, iterative=True, **kwargs):
    if method is None and iterative is True:
        method = default_iterative
    elif method is None and iterative is False:
        raise NotImplementedError(
            "Non-iterative methods for SR are no longer implemented"
        )

    if method == "cg":
        return SR_otf_cg(diag_shift, **kwargs)
    elif method == "gmres":
        return SR_otf_gmres(diag_shift, **kwargs)
    else:
        raise NotImplementedError("Only cg and gmres are implemented")
