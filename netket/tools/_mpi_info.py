from subprocess import CalledProcessError

from ._common import exec_in_terminal


def get_global_mpi_info():
    info = {}
    try:
        info["mpicc"] = exec_in_terminal(["which", "mpicc"])
    except CalledProcessError:
        info["mpicc"] = ""

    try:
        info["mpirun"] = exec_in_terminal(["which", "mpirun"])
    except CalledProcessError:
        info["mpirun"] = ""

    try:
        info["mpiexec"] = exec_in_terminal(["which", "mpiexec"])
    except CalledProcessError:
        info["mpiexec"] = ""

    if info["mpicc"]:
        info["link_flags"] = get_link_flags(exec_in_terminal([info["mpicc"], "-show"]))
    else:
        info["link_flags"] = ""
    return info


def get_link_flags(flags):
    flags = flags.replace(",", " ").split()
    res = []
    for flag in flags:
        if flag.startswith("-L"):
            res.append(flag)

    return res
