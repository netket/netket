from typing import Union, Tuple
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
    return tuple(int(v) for v in verstr.split("."))[:3]


def module_version(module: Union[str, ModuleType]) -> Tuple[int, ...]:
    if isinstance(module, str):
        module = importlib.import_module(module)

    return version_tuple(module.__version__)


def version_string(module: Union[str, ModuleType]) -> str:
    if isinstance(module, str):
        module = importlib.import_module(module)

    return module.__version__
