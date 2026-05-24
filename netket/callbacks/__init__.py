# Copyright 2020, 2021 The NetKet Authors - All rights reserved.
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

from netket._src.callbacks.base import (
    AbstractCallback as AbstractCallback,
    StopRun as StopRun,
)

from netket._src.callbacks.auto_chunk_size import (
    AutoChunkSize as AutoChunkSize,
)
from netket._src.callbacks.auto_slurm_requeue import (
    AutoSlurmRequeue as AutoSlurmRequeue,
)

from netket._src.callbacks.early_stopping import EarlyStopping as EarlyStopping
from netket._src.callbacks.timeout import Timeout as Timeout
from netket._src.callbacks.invalid_loss_stopping import (
    InvalidLossStopping as InvalidLossStopping,
)
from netket._src.callbacks.convergence_stopping import (
    ConvergenceStopping as ConvergenceStopping,
)

from netket.utils import _hide_submodules

_hide_submodules(__name__)
