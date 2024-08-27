# Copyright 2024 The NetKet Authors - All rights reserved.
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
import jax

from .base import ModuleFramework, framework


# expose jax-stax as a flax module
class EquinoxWrapper:
    def __init__(self, static_module, params_treedef):
        self.static_module = static_module
        self.params_treedef = params_treedef

    def init(self, rng, *args, **kwargs):
        raise RuntimeError("not allowed")

    def apply(
        self,
        variables,
        *args,
        rngs=None,
        method=None,  # noqa: W0613
        mutable=False,
        **kwargs,
    ):
        if mutable is not False:
            raise NotImplementedError("Not implemented")
        if rngs is not None:
            raise NotImplementedError()
        if method is None:
            method = "__call__"

        module = self.recompose(variables)
        fun = getattr(module, method)

        return fun(*args, key=rngs, **kwargs)

    def recompose(self, variables):
        import equinox as eqx

        params = list(variables["params"]["list"])

        params_module = jax.tree.unflatten(self.params_treedef, params)
        module = eqx.combine(params_module, self.static_module)

        return module

    def __repr__(self):
        return f"EquinoxWrapper({self.static_module})"


@framework
class EquinoxFramework(ModuleFramework):
    name: str = "Equinox"

    @staticmethod
    def is_loaded() -> bool:
        return "equinox" in sys.modules

    @staticmethod
    def is_my_module(module) -> bool:
        # this will only get called if the module is loaded
        import equinox as eqx  # noqa: E0401

        # jax modules are tuples
        if isinstance(module, eqx.Module):
            return True

        return False

    @staticmethod
    def wrap(module):
        import equinox as eqx

        params, static = eqx.partition(module, eqx.is_array)
        params_list, params_treedef = jax.tree.flatten(params)
        variables = {"params": {"list": tuple(params_list)}}

        return variables, EquinoxWrapper(static, params_treedef)
