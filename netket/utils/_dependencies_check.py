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

"""
This file contains logic to check version of NetKet's dependencies and hard-error in
case of outdated dependencies that we know might break NetKet's numerical results
silently or unexpectedly.
"""

from textwrap import dedent


from .version_check import module_version, version_string


def create_msg(pkg_name, cur_version, desired_version, extra_msg="", pip_pkg_name=None):
    if pip_pkg_name is None:  # pragma: no cover
        pip_pkg_name = pkg_name
    return dedent(
        f"""

        ##########################################################################################

        {pkg_name} version {cur_version} (< {desired_version}) is too old and incompatible with NetKet.
        Please update `{pkg_name}` by running the command:

            pip install --upgrade pip
            pip install --upgrade netket {pip_pkg_name}

        (assuming you are using pip. Similar commands can be used on conda).

        This error most likely happened because you either have an old version of `pip`
        or you are hard-coding the `{pkg_name}` version in a requirements file.

        {extra_msg}

        ##########################################################################################

        """
    )


if not module_version("jax") >= (0, 4, 16):  # pragma: no cover
    cur_version = version_string("jax")
    raise ImportError(create_msg("jax", cur_version, "0.4.16"))


if not module_version("optax") >= (0, 1, 3):  # pragma: no cover
    cur_version = version_string("optax")
    extra = """Reason: Optax is NetKet's provider of optimisers. Versions before 0.1.1 did not
               support complex numbers and silently returned wrong values, especially when
               using optimisers involving the norm of the gradient such as `Adam`.
               As recent versions of optax correctly work with complex numbers, please upgrade.
               """
    raise ImportError(create_msg("optax", cur_version, "0.1.3", extra))

if not module_version("flax") >= (0, 6, 5):  # pragma: no cover
    cur_version = version_string("flax")
    extra = """Reason: Flax is NetKet's default neural-network library. Versions before 0.5 had
               a bug and did not properly support complex numbers.
               """
    raise ImportError(create_msg("flax", cur_version, "0.6.5", extra))

# TODO: Uncomment and bump version once we unvendor plum.
# if not module_version("plum") >= (2, 2, 2):  # pragma: no cover
#     raise ImportError(
#         create_msg(
#             "plum", version_string("plum"), "2.2.2", pip_pkg_name="plum-dispatch"
#         )
#     )
