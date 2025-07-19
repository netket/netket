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

from .legacy.earlystopping import EarlyStopping as EarlyStopping
from .legacy.timeout import Timeout as Timeout
from .legacy.invalidlossstopping import InvalidLossStopping as InvalidLossStopping
from .legacy.convergence_stopping import ConvergenceStopping as ConvergenceStopping

from .base import AbstractCallback as AbstractCallback
from .callback_list import CallbackList as CallbackList
from .legacy_wrappers import (
    LegacyCallbackWrapper as LegacyCallbackWrapper,
    LegacyLoggerWrapper as LegacyLoggerWrapper,
)
from .progressbar import ProgressBarCallback as ProgressBarCallback

from netket.utils import _hide_submodules

_hide_submodules(__name__)
