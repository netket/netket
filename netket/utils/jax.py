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

from typing import Callable

from . import struct


def get_afun_if_module(mod_or_fun) -> Callable:
    """Returns the apply function if it's a module. Does nothing otherwise."""
    if hasattr(mod_or_fun, "apply"):
        return mod_or_fun.apply
    else:
        return mod_or_fun


@struct.dataclass
class WrappedApplyFun:
    """Wraps a callable to be a module-like object with the method `apply`."""

    apply: Callable
    """The wrapped callable."""

    def __repr__(self):
        return f"{type(self).__name__}(apply={self.apply}, hash={hash(self)})"


def wrap_afun(mod_or_fun):
    """Wraps a callable to be a module-like object with the method `apply`.
    Does nothing if it already has an apply method.
    """
    if hasattr(mod_or_fun, "apply"):
        return mod_or_fun
    else:
        return WrappedApplyFun(mod_or_fun)
