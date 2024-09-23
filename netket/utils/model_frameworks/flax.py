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

import sys

from .base import ModuleFramework, framework


@framework
class FlaxFramework(ModuleFramework):
    name: str = "Flax"

    @staticmethod
    def is_loaded() -> bool:
        # this should be not necessary, as netket requires and loads
        # Flax, but let's set a good example
        return "flax" in sys.modules

    @staticmethod
    def is_my_module(module) -> bool:
        # this will only get called if the module is loaded
        from flax import linen as nn

        return isinstance(module, nn.Module)

    @staticmethod
    def wrap(module):
        return None, module
