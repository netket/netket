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

import importlib

__all__ = [
    "driver",
    "dynamics",
    "sampler",
    "vqs",
    "TDVP",
    "models",
    "hilbert",
    "operator",
    "logging",
    "observable",
    "QSR",
]

_submodules = {
    "driver": "netket.experimental.driver",
    "dynamics": "netket.experimental.dynamics",
    "sampler": "netket.experimental.sampler",
    "vqs": "netket.experimental.vqs",
    "models": "netket.experimental.models",
    "hilbert": "netket.experimental.hilbert",
    "operator": "netket.experimental.operator",
    "logging": "netket.experimental.logging",
    "observable": "netket.experimental.observable",
    "qsr": "netket.experimental.qsr",
}


def __getattr__(name):
    if name == "TDVP":
        module = importlib.import_module("netket.experimental.driver")
        val = module.TDVP
    elif name == "QSR":
        module = importlib.import_module("netket.experimental.qsr")
        val = module.QSR
    elif name in _submodules:
        val = importlib.import_module(_submodules[name])
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    globals()[name] = val
    return val


from netket.utils import _hide_submodules

_hide_submodules(__name__)
