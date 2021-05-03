import sys
import os
import platform
import importlib

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
