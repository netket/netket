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

import jax


def get_afun_if_module(mod_or_fun, *args, **kwargs):
    if hasattr(mod_or_fun, "apply"):
        return mod_or_fun.apply
    else:
        return mod_or_fun


class WrappedApplyFun:
    def __init__(self, module):
        self.apply = module
        self._hash = None

    def __eq__(self, other):
        return type(other) is WrappedApplyFun and self.apply == other.apply

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.apply)
        return self._hash


def wrap_afun(mod_or_fun, *args, **kwargs):
    if hasattr(mod_or_fun, "apply"):
        return mod_or_fun
    else:
        return WrappedApplyFun(mod_or_fun)
