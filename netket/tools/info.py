# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import platform

from ._common import version, is_available
from ._cpu_info import cpu_info

PLATFORM = sys.platform

STD_SPACE = 20


def printfmt(key, value=None, *, indent=0, pre="", alignment=STD_SPACE):
    INDENT_STR = "- "

    if indent > 0:
        indent_spaces = indent * 2
        alignment -= indent_spaces + len(INDENT_STR)

        pre = f"{pre}{'' : <{indent_spaces}}{INDENT_STR}"

    if value is not None:
        print(f"{pre}{key : <{alignment}} : {value}", flush=True)
    else:
        print(f"{pre}{key : <{alignment}}", flush=True)


def _fmt_device(dev):
    return f"<{dev.id}: {dev.device_kind}>"


def info():
    print("====================================================")
    print("==         NetKet Diagnostic Information          ==")
    print("====================================================")

    # try to import version without import netket itself
    from .. import _version

    printfmt("NetKet version", _version.version)
    print()

    print("# Python")
    printfmt("implementation", platform.python_implementation(), indent=1)
    printfmt("version", platform.python_version(), indent=1)
    printfmt("distribution", platform.python_compiler(), indent=1)
    printfmt("path", sys.executable, indent=1)
    print()

    # Try to detect platform
    print("# Host information")
    printfmt("System      ", platform.platform(), indent=1)
    printfmt("Architecture", platform.machine(), indent=1)

    # Try to query cpu info
    platform_info = cpu_info()

    printfmt("AVX", platform_info["supports_avx"], indent=1)
    printfmt("AVX2", platform_info["supports_avx2"], indent=1)
    printfmt("Cores", platform_info["core_count"], indent=1)
    print()

    # try to load jax
    print("# NetKet dependencies")
    printfmt("numpy", version("numpy"), indent=1)
    printfmt("jaxlib", version("jaxlib"), indent=1)
    printfmt("jax", version("jax"), indent=1)
    printfmt("equinox", version("equinox"), indent=1)
    printfmt("flax", version("flax"), indent=1)
    printfmt("optax", version("optax"), indent=1)
    printfmt("numba", version("numba"), indent=1)
    printfmt("netket", version("netket"), indent=1)
    print()

    if is_available("jax"):
        print("# Jax ")
        import jax

        backends = _jax_backends()
        printfmt("backends", backends, indent=1)
        for backend in backends:
            printfmt(
                f"{backend}",
                [_fmt_device(dev) for dev in jax.devices(backend)],
                indent=2,
            )
        print()


def _jax_backends():
    import jax

    backends = []
    for backend in ["cpu", "gpu", "tpu"]:
        try:
            jax.devices(backend)
        except RuntimeError:
            pass
        else:
            backends.append(backend)
    return backends


info()
