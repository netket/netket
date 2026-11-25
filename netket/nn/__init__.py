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

from .symmetric_linear import (
    DenseSymm,
    DenseEquivariant,
)

from netket.nn.masked_linear import MaskedDense1D, MaskedConv1D, MaskedConv2D
from netket.nn.fast_masked_linear import (
    FastMaskedDense1D,
    FastMaskedConv1D,
    FastMaskedConv2D,
)

from netket.nn.utils import (
    to_array,
    to_matrix,
    binary_encoding,
)

from netket.nn import blocks as blocks
from netket.nn import activation as activation
from netket.nn import apply_operator as apply_operator

from netket._src.nn.freeze.api import (
    freeze_parameters as freeze_parameters,
    unfreeze_parameters as unfreeze_parameters,
)

# Import for deprecation machinery
from netket._src.nn.apply_operator.functional import (
    make_logpsi_op_afun as _deprecated_make_logpsi_op_afun,
)


# Deprecation machinery
_deprecations = {
    # December 2025, NetKet 3.21
    "make_logpsi_op_afun": (
        "netket.nn.make_logpsi_op_afun is deprecated: use netket.nn.apply_operator.make_logpsi_op_afun",
        _deprecated_make_logpsi_op_afun,
    ),
}


from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

__getattr__ = _deprecation_getattr(__name__, _deprecations)

del _deprecation_getattr
