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

from typing import Optional

from netket.utils import wraps_legacy
from netket.legacy.optimizer import SR as SR_legacy
from netket.legacy.machine import AbstractMachine

from .base import SR
from .sr_onthefly import SRLazyCG, SRLazyGMRES

default_iterative = "cg"
# default_direct = "eigen"


@wraps_legacy(SR_legacy, "machine", AbstractMachine)
def SR(
    diag_shift: float = 0.01,
    method: Optional[str] = None,
    *,
    iterative: bool = True,
    **kwargs,
) -> SR:
    """
    Construct the structure holding the parameters for using the
    Stochastic Reconfiguration/Natural gradient method.

    Depending on the arguments, an implementation is chosen. For
    details on all possible kwargs check the specific SR implementations
    in the documentation.

    You can also construct one of those structures directly.

    Args:
        diag_shift: Diagonal shift added to the S matrix
        method: (cg, gmres) The specific method.
        iterative: Whever to use an iterative method or not.

    Returns:
        The SR parameter structure.
    """
    if method is None and iterative is True:
        method = default_iterative
    elif method is None and iterative is False:
        raise NotImplementedError(
            "Non-iterative methods for SR are no longer implemented"
        )

    if method == "cg":
        return SRLazyCG(diag_shift, **kwargs)
    elif method == "gmres":
        return SRLazyGMRES(diag_shift, **kwargs)
    else:
        raise NotImplementedError("Only cg and gmres are implemented")
