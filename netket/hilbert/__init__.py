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

from . import constraint
from . import index

from .abstract_hilbert import AbstractHilbert
from .discrete_hilbert import DiscreteHilbert
from .homogeneous import HomogeneousHilbert

from .continuous_hilbert import ContinuousHilbert

from .doubled_hilbert import DoubledHilbert
from .spin import Spin
from .fock import Fock
from .qubit import Qubit
from .particle import Particle
from .spin_orbital_fermions import SpinOrbitalFermions

from .tensor_hilbert import TensorHilbert
from . import tensor_hilbert_discrete

from . import random


# Deprecated bindings
from .custom_hilbert import CustomHilbert as _deprecated_CustomHilbert

_deprecations = {
    # September 2024, NetKet 3.14
    "CustomHilbert": (
        "netket.hilbert.CustomHilbert is deprecated: use custom constraints with "
        "existing hilbert spaces instead, or define your own hilbert space class.",
        _deprecated_CustomHilbert,
    ),
}

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr
from netket.utils import _hide_submodules

__getattr__ = _deprecation_getattr(__name__, _deprecations)
_hide_submodules(__name__)

del _deprecation_getattr
