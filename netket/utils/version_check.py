from types import ModuleType

import importlib


def version_tuple(verstr: str):
    # drop everything after the numeric part of the version
    allowed_chars = "0123456789."
    for i, char in enumerate(verstr):
        if char not in allowed_chars:
            break
    else:
        i = len(verstr) + 1

    verstr = verstr[:i].rstrip(".")
    vertupl = tuple(int(v) for v in verstr.split("."))[:3]

    # Ensure that we have (major, minor, patch)
    while len(vertupl) < 3:
        vertupl = vertupl + (0,)
    return vertupl


def module_version(module: str | ModuleType) -> tuple[int, ...]:
    if isinstance(module, str):
        module = importlib.import_module(module)

    return version_tuple(module.__version__)


def version_string(module: str | ModuleType) -> str:
    if isinstance(module, str):
        module = importlib.import_module(module)

    return module.__version__
