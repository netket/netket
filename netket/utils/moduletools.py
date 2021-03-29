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


def _hide_submodules(module_name, *, remove_self=True, ignore=[]):
    """
    Hide all submodules created by files (not folders) in module_name defined
    at module_path.
    If remove_self=True, also removes itself from the module.
    """
    import sys, os

    module = sys.modules[module_name]
    module_path = module.__path__[0]

    for file in os.listdir(module_path):
        if file.endswith(".py") and not file == "__init__.py":
            mod_name = file[:-3]
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


def rename_class(new_name):
    """
    Decorator to renames a class
    """

    def decorator(clz):
        clz.__name__ = new_name
        clz.__qualname__ = new_name
        return clz

    return decorator
