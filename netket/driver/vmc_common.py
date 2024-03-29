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

from netket.utils import deprecated


@deprecated(reason="Will be removed. Please remove usages.")
def _info(obj, depth=None):
    if hasattr(obj, "info") and callable(obj.info):
        return obj.info(depth)
    else:
        return str(obj)


_deprecated_names = {"info": _info}


def __getattr__(name):
    from netket.utils import warn_deprecation

    if name in _deprecated_names:
        warn_deprecation(
            " \n"
            " \n"
            "          =======================================================================\n"
            "            `nk.driver.vmc_common is deprecated and the functionality removed.   \n"
            "          =======================================================================\n"
            " \n"
            "If you imported `nk.driver.vmc_common`, you must reimplement that functionality yourself.\n\n"
        )
        return _deprecated_names[name]

    raise AttributeError(f"module {__name__} has no attribute {name}")
