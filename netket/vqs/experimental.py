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

from netket.utils import warn_deprecation as _warn_deprecation

_old_module_name = "netket.vqs.experimental"
_new_module_name = "netket.experimental.vqs"

_deprecated_names = ["variables_from_file", "variables_from_tar"]


def __getattr__(name):
    from netket.experimental import vqs as _new_module

    if name in _deprecated_names:
        _warn_deprecation(
            f"The `{_old_module_name}` module is deprecated. Use `{_new_module_name}` instead.\n"
            f"To fix this warning, change `{_old_module_name}.{name} to `{_new_module_name}.{name}`."
        )
        return getattr(_new_module, name)

    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    from netket.experimental import vqs as _new_module

    return dir(_new_module)
