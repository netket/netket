import importlib
import sys
from subprocess import CalledProcessError, PIPE, check_output


def exec_in_terminal(command):
    """Run a command in the terminal and get the
    output stripping the last newline.

    Args:
        command: a string or list of strings
    """
    # On Windows, when using `where` to find a command, it will output some
    # message to stderr if the command is not found.
    # We redirect stderr to PIPE to prevent that message from showing on the screen.
    return check_output(command, stderr=PIPE).strip().decode("utf8")


def is_available(lib_name: str) -> bool:
    """
    Checks if a library can be imported
    """
    try:
        importlib.import_module(lib_name)
        available = True
    except ImportError:
        available = False

    return available


def version(lib_name) -> str:
    """
    Returns the version of a library as a string or
    unavailable if it cannot be imported
    """
    if is_available(lib_name):
        return _version(lib_name)
    else:
        return "unavailable"


def _version(lib_name):
    """
    Returns the version of a package.
    If version cannot be determined returns "available"
    """
    lib = importlib.import_module(lib_name)
    if hasattr(lib, "__version__"):
        return lib.__version__
    else:
        return "available"


def get_executable_path(name):
    """
    Get the path of an executable.
    """
    # I. Hate. Windows.
    if sys.platform.startswith("win32"):
        os_which = "where"
    else:
        os_which = "which"

    try:
        path = exec_in_terminal([os_which, name])
    except (CalledProcessError, FileNotFoundError):
        path = ""
    return path
