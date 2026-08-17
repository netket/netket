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

from netket.utils.version_check import version_tuple


def import_optional_dependency(
    name: str, minimum_version="", maximum_version="", descr=""
) -> ModuleType:
    """Try to import library `name`, and if it cannot be found or its version is
    outside of the supported range, raise an informative error.

    Args:
        name: the name of the module to import.
        minimum_version: if specified, the oldest supported version (inclusive).
        maximum_version: if specified, the first unsupported version (exclusive).
        descr: description of the NetKet functionality requiring this module,
            used in the error messages.
    """
    bounds = []
    if minimum_version != "":
        bounds.append(f">={minimum_version}")
    if maximum_version != "":
        bounds.append(f"<{maximum_version}")
    requirement = name + ",".join(bounds)

    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"""

            Could not import `{name}`, which is necessary to use
            `{descr}`.

            To install it, run

                pip install '{requirement}'

            """
        )

    # If the version cannot be determined, optimistically assume it is fine.
    version = getattr(module, "__version__", None)
    if bounds and version is not None:
        installed = version_tuple(version)
        if (minimum_version != "" and installed < version_tuple(minimum_version)) or (
            maximum_version != "" and installed >= version_tuple(maximum_version)
        ):
            raise ImportError(
                f"""

                `{name}` version {version} is not supported by
                `{descr}`, which requires `{requirement}`.

                To install a supported version, run

                    pip install '{requirement}'

                """
            )

    return module
