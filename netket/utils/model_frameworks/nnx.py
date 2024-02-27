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

from typing import TYPE_CHECKING
from functools import partial

import sys
import jax

from .base import ModuleFramework, framework

if TYPE_CHECKING:
    from flax import nnx


# expose jax-stax as a flax module
class NNXWrapper:
    def __init__(self, graphdef, params_structdef, model_state_structdef):
        self.graphdef = graphdef
        self.params_structdef = params_structdef
        self.model_state_structdef = model_state_structdef

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

        return fun(*args, **kwargs)

    def recompose(self, variables):
        from flax import nnx

        model_state_leaves = variables["model_state"]
        params_leaves = variables["params"]["leaves"]

        model_state = jax.tree.unflatten(self.model_state_structdef, model_state_leaves)
        params = jax.tree.unflatten(self.params_structdef, params_leaves)

        nnx_module = nnx.merge(self.graphdef, params, model_state)
        return nnx_module

    def __getattr__(self, name):
        if hasattr(self.graphdef.type, name):
            return partial(self.apply, method=name)
        raise AttributeError(
            f"'{type(self).__name__}' (and the wrapped '{self.graphdef.type}') object has no attribute '{name}'"
        )

    def __repr__(self):
        return f"NNXWrapper(wrapped_class={self.graphdef.type}, ...)"


@framework
class NNXFramework(ModuleFramework):
    name: str = "NNX"

    @staticmethod
    def is_loaded() -> bool:
        # this should be not necessary, as netket requires and loads
        # Flax, but let's set a good example
        return "flax" in sys.modules and (
            "flax.experimental.nnx" in sys.modules or "flax.nnx" in sys.modules
        )

    @staticmethod
    def is_my_module(module) -> bool:
        # this will only get called if the module is loaded
        from flax import nnx

        return isinstance(module, nnx.Module)

    @staticmethod
    def wrap(module):
        from flax import nnx

        graphdef, params, model_state = nnx.split(module, nnx.Param, ...)
        params_leaves, params_structdef = jax.tree.flatten(params)
        model_state_leaves, model_state_structdef = jax.tree.flatten(model_state)

        variables = {
            "model_state": tuple(model_state_leaves),
            "params": {"leaves": tuple(params_leaves)},
        }

        return variables, NNXWrapper(graphdef, params_structdef, model_state_structdef)

    @staticmethod
    def unwrap(module, maybe_variables) -> "nnx.Module":
        from flax import nnx

        model_state_leaves = maybe_variables["model_state"]
        params_leaves = maybe_variables["params"]["leaves"]

        model_state = jax.tree.unflatten(
            module.model_state_structdef, model_state_leaves
        )
        params = jax.tree.unflatten(module.params_structdef, params_leaves)

        nnx_module = nnx.merge(module.graphdef, params, model_state)
        return nnx_module
