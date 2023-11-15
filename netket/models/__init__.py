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

from .rbm import RBM, RBMModPhase, RBMMultiVal, RBMSymm
from .equivariant import GCNN
from .full_space import LogStateVector
from .jastrow import Jastrow
from .mps import MPSPeriodic
from .gaussian import Gaussian
from .deepset import DeepSetRelDistance, DeepSetMLP
from .ndm import NDM
from .autoreg import (
    AbstractARNN,
    ARNNSequential,
    ARNNDense,
    ARNNConv1D,
    ARNNConv2D as _ARNNConv2D,
)
from .fast_autoreg import (
    FastARNNSequential,
    FastARNNDense,
    FastARNNConv1D,
    FastARNNConv2D as _FastARNNConv2D,
)
from .mlp import MLP

from .utils import update_GCNN_parity


# TODO: remove dtype deprecation
from netket.utils import deprecate_dtype as _deprecate_dtype

ARNNConv2D = _deprecate_dtype(_ARNNConv2D)
FastARNNConv2D = _deprecate_dtype(_FastARNNConv2D)


from netket.utils import _hide_submodules

_hide_submodules(__name__)
