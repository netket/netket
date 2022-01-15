# Copyright 2020, 2021 The NetKet Authors - All rights reserved.
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

# Deprecation warnings for netket.variational, renamed to netket.vqs

from netket.utils import warn_deprecation as _warn_deprecation

from . import vqs as _new_module

_old_module_name = "netket.variational"
_new_module_name = "netket.vqs"

_deprecated_names = [name for name in dir(_new_module) if not name.startswith("_")]


def __getattr__(name):
    if name in _deprecated_names:
        _warn_deprecation(
            f"The `{_old_module_name}` module is deprecated. Use `{_new_module_name}` instead.\n"
            f"To fix this warning, change `{_old_module_name}.{name} to `{_new_module_name}.{name}`."
        )
        return getattr(_new_module, name)

    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return dir(_new_module)
