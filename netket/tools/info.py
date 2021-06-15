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

from ._common import exec_in_terminal, version, is_available
from ._cpu_info import cpu_info
from ._mpi_info import get_global_mpi_info, get_link_flags

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


def info():
    """
    When called via::

        # python3 -m netket.tools.check_mpi
        mpi_available     : True
        mpi4jax_available : True
        n_nodes           : 1

    this will print out basic MPI information to make allow users to check whether
    the environment has been set up correctly.
    """
    print("====================================================")
    print("==         NetKet Diagnostic Informations         ==")
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
    print("# Host informations")
    printfmt("System      ", platform.platform(), indent=1)
    printfmt("Architecture", platform.machine(), indent=1)

    # Try to query cpu info
    platform_info = cpu_info()

    printfmt("AVX", platform_info["supports_avx"], indent=1)
    printfmt("AVX2", platform_info["supports_avx2"], indent=1)
    if "cpu cores" in platform_info:
        printfmt("Cores", platform_info["cpu cores"], indent=1)
    elif "cpu_cores" in platform_info:
        printfmt("Cores", platform_info["cpu_cores"], indent=1)
    elif "core_count" in platform_info:
        printfmt("Cores", platform_info["core_count"], indent=1)
    print()

    # try to load jax
    print("# NetKet dependencies")
    printfmt("numpy", version("numpy"), indent=1)
    printfmt("jaxlib", version("jaxlib"), indent=1)
    printfmt("jax", version("jax"), indent=1)
    printfmt("flax", version("flax"), indent=1)
    printfmt("optax", version("optax"), indent=1)
    printfmt("numba", version("numba"), indent=1)
    printfmt("mpi4py", version("mpi4py"), indent=1)
    printfmt("mpi4jax", version("mpi4jax"), indent=1)
    printfmt("netket", version("netket"), indent=1)
    print()

    if is_available("jax"):
        print("# Jax ")
        import jax

        backends = _jax_backends()
        printfmt("backends", backends, indent=1)
        for backend in backends:
            printfmt(f"{backend}", jax.devices(backend), indent=2)
        print()

    if is_available("mpi4jax"):
        print("# MPI4JAX")
        import mpi4jax

        if hasattr(mpi4jax._src.xla_bridge, "HAS_GPU_EXT"):
            printfmt("HAS_GPU_EXT", mpi4jax._src.xla_bridge.HAS_GPU_EXT, indent=1)
        print()

    if is_available("mpi4py"):
        print("# MPI ")
        import mpi4py
        from mpi4py import MPI

        printfmt("mpi4py", indent=1)
        printfmt("MPICC", mpi4py.get_config()["mpicc"], indent=1)
        printfmt(
            "MPI link flags",
            get_link_flags(exec_in_terminal([mpi4py.get_config()["mpicc"], "-show"])),
            indent=1,
        )
        printfmt("MPI version", MPI.Get_version(), indent=2)
        printfmt("MPI library_version", MPI.Get_library_version(), indent=2)

        global_info = get_global_mpi_info()
        printfmt("global", indent=1)
        printfmt("MPICC", global_info["mpicc"], indent=2)
        printfmt("MPI link flags", global_info["link_flags"], indent=2)
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
