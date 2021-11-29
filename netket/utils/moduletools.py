# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys


def _hide_submodules(
    module_name, *, remove_self=True, ignore=tuple(), hide_folder=tuple()
):
    """
    Hide all submodules created by files (not folders) in module_name defined
    at module_path.
    If remove_self=True, also removes itself from the module.
    """
    import os

    module = sys.modules[module_name]
    module_path = module.__path__[0]

    for file in os.listdir(module_path):
        if file.endswith(".py") and not file == "__init__.py":
            mod_name = file[:-3]
        elif file in hide_folder:
            mod_name = file
        else:
            mod_name = None

        if mod_name is not None:
            if (
                hasattr(module, mod_name)
                and mod_name[0] != "_"
                and mod_name not in ignore
            ):
                new_name = "_" + mod_name
                setattr(module, new_name, getattr(module, mod_name))
                delattr(module, mod_name)

    if remove_self and hasattr(module, "_hide_submodules"):
        delattr(module, "_hide_submodules")

    auto_export(module)


def rename_class(new_name):
    """
    Decorator to renames a class
    """

    def decorator(clz):
        clz.__name__ = new_name
        clz.__qualname__ = new_name
        return clz

    return decorator


def export(fn):
    """
    Add the function `fn` to the list of exported attributes of this
    module, `__all__`.

    Args:
        fn: the function or class to export.
    """
    mod = sys.modules[fn.__module__]
    if hasattr(mod, "__all__"):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def auto_export(module):
    """
    Automatically construct __all__ with all modules desired.
    This is necessary to have correct paths in the documentation.

    Args:
        module: a module or module name
    """
    if isinstance(module, str):
        module = sys.modules[module]

    elements = dir(module)

    if not hasattr(module, "__all__"):
        setattr(module, "__all__", [])

    _all = module.__all__

    for el in elements:
        if el.startswith("_"):
            continue

        if el not in _all:
            _all.append(el)


def hide_unexported(module_name):
    """
    Overloads the `__dir__` function of the given module in order to
    only show on autocompletion the attributes inside of `__all__`.

    You can add to `__all__` by using the decorator :ref:`@export`.

    Args:
        module_name: the name of the module to process.
    """
    module = sys.modules[module_name]

    def __dir__():
        return module.__all__

    setattr(module, "__dir__", __dir__)
