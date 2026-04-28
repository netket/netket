# Copyright 2026 The NetKet Authors - All rights reserved.
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

from netket.utils.model_frameworks.base import ModuleFramework, framework


@framework
class BoundFlaxFramework(ModuleFramework):
    name: str = "BoundFlax"

    @property
    def model_contains_parameters(self) -> bool:
        """
        Returns True if the model contains the parameters in the model itself, False
        if the parameters are stored separately.
        """
        return True

    @staticmethod
    def is_loaded() -> bool:
        return "flax" in sys.modules

    @staticmethod
    def is_my_module(module) -> bool:
        from flax import linen as nn

        return (
            isinstance(module, nn.Module) and getattr(module, "scope", None) is not None
        )

    @staticmethod
    def wrap(module):
        unbound_module, variables = module.unbind()
        return variables, unbound_module

    @staticmethod
    def unwrap(wrapped_module, maybe_variables):
        return wrapped_module.bind(maybe_variables)
