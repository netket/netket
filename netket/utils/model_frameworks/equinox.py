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

from typing import TYPE_CHECKING

import sys
import jax

from .base import ModuleFramework, framework

if TYPE_CHECKING:
    import equinox


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

    @property
    def model_contains_parameters(self) -> bool:
        """
        Returns True if the model contains the parameters in the model itself, False
        if the parameters are stored separately.
        """
        return True

    @staticmethod
    def is_loaded() -> bool:
        return "equinox" in sys.modules

    @staticmethod
    def is_my_module(module) -> bool:
        # this will only get called if the module is loaded
        import equinox as eqx  # noqa: E0401

        return isinstance(module, eqx.Module)

    @staticmethod
    def wrap(module: "equinox.Module") -> tuple[dict, EquinoxWrapper]:
        import equinox as eqx

        params, static = eqx.partition(module, eqx.is_array)
        params_leaves, params_treedef = jax.tree.flatten(params)
        variables = {"params": {"list": tuple(params_leaves)}}

        return variables, EquinoxWrapper(static, params_treedef)

    @staticmethod
    def unwrap(
        wrapped_module: EquinoxWrapper, maybe_variables: dict
    ) -> "equinox.Module":
        import equinox as eqx

        params_leaves = maybe_variables["params"]["list"]
        params_module = jax.tree.unflatten(wrapped_module.params_treedef, params_leaves)

        eqx_module = eqx.combine(params_module, wrapped_module.static_module)
        return eqx_module
