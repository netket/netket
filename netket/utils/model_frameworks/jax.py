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

import flax
from flax.core import freeze

from .base import ModuleFramework, framework

# expose jax-stax as a flax module
class JaxWrapper:
    def __init__(self, ifun, afun):
        self.ifun = ifun
        self.afun = afun

    def init(self, keys, inpt):
        _, variables = self.ifun(keys["params"], inpt.shape)
        return freeze({"params": variables})

    def apply(self, variables, *args, rngs=None, method=None, mutable=False, **kwargs):
        if mutable is not False:
            raise ValueError("A wrapped jax module cannot be mutable")

        return self.afun(variables["params"], *args, **kwargs)

    def unwrap_params(self, variables):
        return variables["params"]


@framework
class JaxFramework(ModuleFramework):

    name: str = "Jax"

    @staticmethod
    def is_loaded() -> bool:
        # this should be not necessary, as netket requires and loads
        # Flax, but let's set a good example
        return "jax" in sys.modules

    @staticmethod
    def is_my_module(module) -> bool:
        # this will only get callede if the module is loaded
        import jax, inspect

        # jax modules are tuples
        if isinstance(module, tuple):
            #  with two elements
            if len(module) == 2:
                ifun, afun = module
                #  and both are functions
                if callable(ifun) and callable(afun):
                    ifun_signature = inspect.getfullargspec(ifun)
                    afun_signature = inspect.getfullargspec(afun)

                    # With two arguments eacch
                    if len(ifun_signature.args) == 2 and len(afun_signature.args) == 2:
                        return True

        return False

    @staticmethod
    def wrap(module):
        return JaxWrapper(*module)

    @staticmethod
    def wrap_params(variables):
        return freeze({"params": variables})

    @staticmethod
    def unwrap_params(variables):
        return variables["params"]
