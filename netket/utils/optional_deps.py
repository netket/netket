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
import importlib.metadata

from netket.utils.version_check import version_tuple


def _version_spec(minimum_version: str, maximum_version: str) -> str:
    """Format the version bounds as a pip requirement specifier."""
    bounds = []
    if minimum_version != "":
        bounds.append(f">={minimum_version}")
    if maximum_version != "":
        bounds.append(f"<{maximum_version}")
    return ",".join(bounds)


def _installed_version(module: ModuleType, name: str) -> str | None:
    """Version of an imported module, or None if it cannot be determined."""
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    return version


def import_optional_dependency(
    name: str, *, minimum_version="", maximum_version="", descr="", extra_msg=""
) -> ModuleType:
    """Try to import library `name`, and if it cannot be found or its version is
    not supported, raise an informative error.

    Args:
        name: the name of the module to import.
        minimum_version: if specified, the oldest supported version (inclusive).
        maximum_version: if specified, the first unsupported version (exclusive).
        descr: description of the NetKet functionality requiring this module,
            used in the error messages.
        extra_msg: additional explanation appended to the error raised when the
            installed version is not supported.
    """
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"""

            Could not import `{name}`, which is necessary to use
            `{descr}`.

            To install it, run

                pip install '{name}{_version_spec(minimum_version, maximum_version)}'

            """
        )

    if minimum_version != "" or maximum_version != "":
        version = _installed_version(module, name)
        # If the version cannot be determined we cannot do anything, so we
        # optimistically assume it is fine.
        if version is not None:
            too_old = minimum_version != "" and version_tuple(version) < version_tuple(
                minimum_version
            )
            too_new = maximum_version != "" and version_tuple(version) >= version_tuple(
                maximum_version
            )
            if too_old or too_new:
                raise ImportError(
                    f"""

                    `{name}` version {version} is not supported by
                    `{descr}`, which requires `{name}{_version_spec(minimum_version, maximum_version)}`.

                    To install a supported version, run

                        pip install '{name}{_version_spec(minimum_version, maximum_version)}'

                    {extra_msg}
                    """
                )

    return module
