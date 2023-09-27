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

from types import ModuleType
import importlib


def import_optional_dependency(name: str, minimum_version="", descr="") -> ModuleType:
    """Try to import library `name`, and if it cannot be found, raise an
    informative error.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        if minimum_version != "":
            minimum_version = f">= {minimum_version}"
        raise ModuleNotFoundError(
            f"""

            Could not import `{name}`, which is necessary to use
            `{descr}`.

            To install it, run

                pip install {name} {minimum_version}

            """
        )
