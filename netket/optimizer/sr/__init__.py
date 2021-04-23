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

from .api import SR

from .base import SMatrix

from .sr_onthefly import SRLazyCG, SRLazyGMRES
from .s_onthefly_mat import SRLazy, LazySMatrix

from .sr_snr import SRSNR

from netket.utils import _hide_submodules

_hide_submodules(__name__)
