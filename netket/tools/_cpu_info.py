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


def cpu_info():
    if PLATFORM.startswith("darwin"):
        return get_sysctl_cpu()
    elif PLATFORM.startswith("linux"):
        return get_proc_cpuinfo()
    elif PLATFORM.startswith("win32"):
        return get_wmic_cpu()
    else:
        raise ValueError(f"Unsupported platform {PLATFORM}")


SYSCTL_KEY_TRANSLATIONS = {
    "model": "model_display",
    "family": "family_display",
    "extmodel": "extended_model",
    "extfamily": "extended_family",
}


SYSCTL_FLAG_TRANSLATIONS = {
    "sse4.1": "sse4_1",
    "sse4.2": "sse4_2",
}


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
    "cpu cores": "core_count",
    "cpu_cores": "core_count",
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


WMIC_KEY_TRANSLATIONS = {
    "manufacturer": "vendor",
    "model": "model_display",
    "level": "family_display",
    "name": "brand",
    "numberofcores": "core_count",
}


def get_wmic_cpu():
    """Get CPU parameters using ``wmic`` Windows utility
    For a description of each CPU field, see:
    https://msdn.microsoft.com/en-us/library/aa394373(v=vs.85).aspx
    """
    wmic_text = exec_in_terminal(
        ["wmic", "cpu", "get", "/all", "/format:textvaluelist"]
    )
    info = {}
    for line in wmic_text.splitlines():
        line = line.strip()
        if line == "":
            continue
        key, value = line.split("=", 1)
        key = key.lower()
        key = WMIC_KEY_TRANSLATIONS.get(key, key)
        try:
            value = int(value)
        except ValueError:
            pass
        if key in info:  # Now we're looking at another processor
            break
        info[key] = value

    # Stepping sometines the empty string in wmic output
    if "stepping" in info and info["stepping"] == "":
        info["stepping"] = None

    # Get extra information from kernel32
    from ctypes import windll, wintypes

    has_feature = windll.kernel32.IsProcessorFeaturePresent
    has_feature.argtypes = [wintypes.DWORD]
    info["flags"] = {
        "sse": has_feature(6),
        "sse2": has_feature(10),
        "sse3": has_feature(13),  # Not available on XP
        "mmx": has_feature(3),
        "3dnow": has_feature(7),
    }
    info["unknown_flags"] = ("ssse3", "sse4_1", "sse4_2")

    # TODO: `IsProcessorFeaturePresent` cannot detect AVX.
    # Maybe we can use another package like `cpuinfo` for this.
    info["supports_avx"] = "Unknown"
    info["supports_avx2"] = "Unknown"

    return info
