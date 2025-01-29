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

from typing import Any

import dataclasses
import abc

PyTree = Any


@dataclasses.dataclass(frozen=True)
class ModuleFramework(abc.ABC):

    @property
    def model_contains_parameters(self) -> bool:
        """
        Returns True if the model contains the parameters in the model itself, False
        if the parameters are stored separately.
        """
        return False

    @staticmethod
    @abc.abstractmethod
    def is_loaded() -> bool:
        """
        Returns True if this module framework has already been loaded by the user.
        """

    @staticmethod
    @abc.abstractmethod
    def is_my_module(module: Any) -> bool:
        """
        Returns True if the given module is from this framework, False otherwise.

        Args:
            module: a module from an unknown framework.
        """

    @staticmethod
    @abc.abstractmethod
    def wrap(module: Any) -> tuple[PyTree | None, Any]:
        """
        Wraps the given module in a way that it behaves like a flax module, possibly
        returning the parameters as well.

        For flax-like modules, this should return None and the module itself. For
        modules that store the parameters in the module itself, it should return the
        parameters and a static object that can be used to apply the module.

        Args:
            A module from the framework corresponding to this class.

        Returns:
            A tuple with the parameters, if any, and the static module.
        """
        raise NotImplementedError

    @staticmethod
    def unwrap(module: Any, maybe_variables: PyTree | None) -> Any:
        """
        Undoes the wrapping done by `wrap`, restoring the original module.

        For flax-like modules, this should do nothing. For modules that store
        the parameters in the module itself, it should unwrap the module and
        restore the parameters.

        Args:
            module: the module to unwrap
            maybe_variables: the variables obtained from the wrapping, if any.

        Returns:
            The original module from this framework.
        """
        return module


registered_frameworks = []


def framework(clz):
    """
    Registers a framework and it's wrapper methods to make it
    behave like a flax framework.
    """
    clz = dataclasses.dataclass(frozen=True)(clz)
    registered_frameworks.append(clz)
    return clz


@dataclasses.dataclass(frozen=True)
class UnknownFramework(ModuleFramework):
    name: str = "Unknown"

    @staticmethod
    def is_loaded() -> bool:
        return True

    @staticmethod
    def is_my_module(module) -> bool:
        return False

    @staticmethod
    def wrap(module) -> tuple:
        return None, module

    @staticmethod
    def unwrap(wrapped_module, wrapped_variables):
        return wrapped_module


def identify_framework(module):
    for _framework in registered_frameworks:
        if _framework.is_loaded() and _framework.is_my_module(module):
            return _framework

    return UnknownFramework


def maybe_wrap_module(module) -> tuple:
    """
    Passing a module from an unknown framework (might be user defined module, a jax
    module, flax or haiku or anything else really), attempt to identify what is the
    package/framework it comes from, and if so it correctly wraps it in order to
    make it behave like a flax module (our default).

    Also returns a function used to unpack the parameters once we are done.
    """
    framewrk = identify_framework(module)

    maybe_module_variables, static_module = framewrk.wrap(module)

    return maybe_module_variables, static_module
