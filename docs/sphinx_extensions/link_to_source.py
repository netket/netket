import os
import inspect
import importlib
import sys


def strip_leading_path(target_path, root="../../"):
    """
    Get relative path.
    Defaults to this file being in a sphinx_ext/
    """
    # Get the current file's directory path
    current_file_dir = os.path.dirname(os.path.abspath(__file__) + root)

    # Compute the relative path from the current file's directory to the target path
    relative_path = os.path.relpath(target_path, current_file_dir)

    # if it's installed in site-packages, strip from there...
    # this happens on readthedocs
    if "site-packages" in relative_path:
        relative_path = relative_path.split("site-packages/")[-1]

    # Remove leading '../' from the relative path
    normalized_path = os.path.normpath(relative_path)
    if normalized_path.startswith("../"):
        normalized_path = normalized_path.lstrip("../")

    return normalized_path


def get_object_from_path(path):
    parts = path.split(".")

    # Try to import the module step by step
    module = None
    for i in range(len(parts), 0, -1):
        try:
            module_name = ".".join(parts[:i])
            module = importlib.import_module(module_name)
            break
        except ImportError:
            continue

    if module is None:
        raise ImportError(f"Could not import any module part from path: {path}")

    obj = module
    for part in parts[i:]:
        obj = getattr(obj, part)

    return obj


def _safe_getfile(function):
    """Safer version of inspect.getfile.

    Classes defined in PyBind, even if they pretend hard
    to be functions, like `jax.jit(fun)`, will raise an error
    if passed to `inspect.getfile`.

    This function will catch those errors and try harder to get
    the underlying file, and raise an error only if it can't.

    This function can only raise an OSError.

    Args:
        function: an object

    Returns:
        A file path.

    Raises:
        OSError
    """
    if isinstance(function, property):
        function = function.fget

    try:
        f_path = inspect.getfile(function)
    except TypeError:  # pragma: no cover
        # raised when the function passed is a C-defined class
        # Check if it contains a module anyway
        if hasattr(function, "__module__"):
            module = sys.modules.get(function.__module__)
            if getattr(module, "__file__", None):
                return module.__file__
            if object.__module__ == "__main__":
                # TODO: properly extract some information from here as well
                raise OSError("source code not available") from None
        raise OSError("source code not available") from None

    return f_path


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to the source code.
    """
    if domain != "py":
        return None
    if not info["module"]:
        return None
    if not info["fullname"]:
        return None

    try:
        obj = get_object_from_path(info["module"] + "." + info["fullname"])
        file = _safe_getfile(obj)
        filename = strip_leading_path(file)
        f_line = inspect.getsourcelines(obj)[1]

        res = f"https://github.com/netket/netket/blob/master/{filename}"
        if f_line is not None:
            res = f"{res}#L{f_line}"

        return res
    except Exception:
        return None


# Adjust the URL to match your repository structure and branch name.
