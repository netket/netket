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

from .rnn import RNN, LSTMNet, GRUNet1D
from .fast_rnn import FastRNN, FastLSTMNet, FastGRUNet1D

from netket.models import Slater2nd as _deprecated_Slater2nd
from netket.models import MultiSlater2nd as _deprecated_MultiSlater2nd

_deprecations = {
    # May 2024
    "Slater2nd": (
        "netket.experimental.models.Slater2nd is deprecated: use "
        "netket.models.Slater2nd (netket >= 3.12)",
        _deprecated_Slater2nd,
    ),
    "MultiSlater2nd": (
        "netket.experimental.models.MultiSlater2nd is deprecated: use "
        "netket.models.MultiSlater2nd (netket >= 3.12)",
        _deprecated_MultiSlater2nd,
    ),
}

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
