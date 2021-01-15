def _hide_submodules(module_name, *, remove_self=True):
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
            if hasattr(module, mod_name):
                new_name = "_" + mod_name
                setattr(module, new_name, getattr(module, mod_name))
                delattr(module, mod_name)

    if remove_self and hasattr(module, "_hide_submodules"):
        delattr(module, "_hide_submodules")


def rename(new_name):
    """
    Decorator to renames a class
    """

    def decorator(clz):
        clz.__name__ = new_name
        clz.__qualname__ = new_name
        return clz

    return decorator
