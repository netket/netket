import sys
import re

from ._common import exec_in_terminal

PLATFORM = sys.platform


def available_cpus():
    """
    Detects the number of logical CPUs subscriptable by this process.
    On Linux, this checks /proc/self/status for limits set by
    taskset, on other platforms taskset do not exist so simply uses
    multiprocessing.

    This should be a good estimate of how many cpu cores Jax/XLA sees.
    """
    if PLATFORM.startswith("linux"):
        try:
            m = re.search(
                r"(?m)^Cpus_allowed:\s*(.*)$", open("/proc/self/status").read()
            )
            if m:
                res = bin(int(m.group(1).replace(",", ""), 16)).count("1")
                if res > 0:
                    return res
        except IOError:
            pass
    else:
        import multiprocessing

        return multiprocessing.cpu_count()


## Taken from https://github.com/matthew-brett/x86cpu/blob/530f89f678aba501381e94be5325b9c91b878a32/x86cpu/tests/info_getters.py#L24
##

SYSCTL_KEY_TRANSLATIONS = dict(
    model="model_display",
    family="family_display",
    extmodel="extended_model",
    extfamily="extended_family",
)


SYSCTL_FLAG_TRANSLATIONS = {
    "sse4.1": "sse4_1",
    "sse4.2": "sse4_2",
}


def cpu_info():
    if PLATFORM.startswith("darwin"):
        return get_sysctl_cpu()
    elif PLATFORM.startswith("linux"):
        return get_proc_cpuinfo()
    else:
        raise ValueError(f"Unsupported platform {PLATFORM}")


def get_sysctl_cpu():
    sysctl_text = exec_in_terminal(["sysctl", "-a"])
    info = {}
    for line in sysctl_text.splitlines():
        if not line.startswith("machdep.cpu."):
            continue
        line = line.strip()[len("machdep.cpu.") :]
        key, value = line.split(": ", 1)
        key = SYSCTL_KEY_TRANSLATIONS.get(key, key)
        try:
            value = int(value)
        except ValueError:
            pass
        info[key] = value
    flags = [flag.lower() for flag in info["features"].split()]
    info["flags"] = [SYSCTL_FLAG_TRANSLATIONS.get(flag, flag) for flag in flags]
    info["unknown_flags"] = ["3dnow"]
    info["supports_avx"] = "hw.optional.avx1_0: 1\n" in sysctl_text
    info["supports_avx2"] = "hw.optionxwal.avx2_0: 1\n" in sysctl_text
    return info


PCPUINFO_KEY_TRANSLATIONS = {
    "vendor_id": "vendor",
    "model": "model_display",
    "family": "family_display",
    "model name": "brand",
}


def get_proc_cpuinfo():
    with open("/proc/cpuinfo", "rt") as fobj:
        pci_lines = fobj.readlines()
    info = {}
    for line in pci_lines:
        line = line.strip()
        if line == "":  # End of first processor
            break
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        key = PCPUINFO_KEY_TRANSLATIONS.get(key, key)
        try:
            value = int(value)
        except ValueError:
            pass
        info[key] = value
    info["flags"] = info["flags"].split()
    # cpuinfo records presence of Prescott New Instructions, Intel's code name
    # for SSE3.
    if "pni" in info["flags"]:
        info["flags"].append("sse3")
    info["unknown_flags"] = ["3dnow"]
    info["supports_avx"] = "avx" in info["flags"]
    info["supports_avx2"] = "avx2" in info["flags"]
    return info
