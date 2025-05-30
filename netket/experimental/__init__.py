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

__all__ = [
    "driver",
    "dynamics",
    "sampler",
    "vqs",
    "TDVP",
    "models",
    "hilbert",
    "geometry",
    "operator",
    "logging",
    "observable",
]

import importlib
import sys

from . import hilbert
from . import geometry


def __getattr__(name):
    if name in {
        "driver",
        "dynamics",
        "sampler",
        "vqs",
        "models",
        "operator",
        "logging",
        "observable",
        "qsr",
    }:
        module = importlib.import_module(f"{__name__}.{name}")
        setattr(sys.modules[__name__], name, module)
        return module
    if name == "TDVP":
        from .driver import TDVP as _TDVP

        setattr(sys.modules[__name__], "TDVP", _TDVP)
        return _TDVP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from netket.utils import _hide_submodules

_hide_submodules(__name__)
