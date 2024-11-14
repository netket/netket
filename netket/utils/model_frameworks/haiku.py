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

from flax import serialization

from .base import ModuleFramework, framework


# expose jax-stax as a flax module
class HaikuWrapper:
    def __init__(self, transformed):
        self.transformed = transformed

    def init(self, rng, *args, **kwargs):
        variables = self.transformed.init(rng["params"], *args, **kwargs)
        return {"params": variables}

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
            raise ValueError("Not implemented")

        return self.transformed.apply(variables["params"], rngs, *args, **kwargs)

    def unwrap_params(self, variables):
        return variables["params"]

    def __repr__(self):
        return f"HaikuWrapper({self.transformed})"


@framework
class HaikuFramework(ModuleFramework):
    name: str = "Haiku"

    @staticmethod
    def is_loaded() -> bool:
        # this should be not necessary, as netket requires and loads
        # Flax, but let's set a good example
        return "haiku" in sys.modules

    @staticmethod
    def is_my_module(module) -> bool:
        # this will only get called if the module is loaded
        import haiku  # noqa: E0401

        # jax modules are tuples
        if isinstance(module, haiku.Transformed):
            return True

        return False

    @staticmethod
    def wrap(module):
        register_serialization_functions()
        return None, HaikuWrapper(module)


already_registered = False


# Haiku uses FlatMapping objects instead of FrozenDict when freezing dicts.
# They are functionally equivalent but we must teach flax how to serialize them.
def register_serialization_functions():
    global already_registered  # noqa: W0603
    if not already_registered:
        already_registered = True
        import haiku  # noqa: E0401

        FlatMappingType = type(haiku.data_structures.to_immutable_dict({"ciao": 1}))

        def serialize_flat_mapping(flat_mapping):
            return dict(flat_mapping)

        def deserialize_flat_mapping(flat_mapping, _):
            return haiku.data_structures.to_immutable_dict(flat_mapping)

        serialization.register_serialization_state(
            FlatMappingType,
            serialize_flat_mapping,
            deserialize_flat_mapping,
        )
