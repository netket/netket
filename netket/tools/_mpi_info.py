from ._common import exec_in_terminal, get_executable_path


def get_global_mpi_info():
    info = {}
    info["mpicc"] = get_executable_path("mpicc")
    info["mpirun"] = get_executable_path("mpirun")
    info["mpiexec"] = get_executable_path("mpiexec")

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
