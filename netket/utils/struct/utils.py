import sys
import builtins

from dataclasses import MISSING


## STUFF FROM python/lib/dataclasses.py
def _set_new_attribute(cls, name, value):
    # Never overwrites an existing attribute.  Returns True if the
    # attribute already exists.
    if name in cls.__dict__:
        return True
    setattr(cls, name, value)
    return False


def _create_fn(
    name, args, body, *, globals=None, locals=None, return_type=MISSING, doc=None
):
    # Note that we mutate locals when exec() is called.  Caller
    # beware!  The only callers are internal to this module, so no
    # worries about external callers.
    if locals is None:
        locals = {}
    if "BUILTINS" not in locals:
        locals["BUILTINS"] = builtins
    return_annotation = ""
    if return_type is not MISSING:
        locals["_return_type"] = return_type
        return_annotation = "->_return_type"
    args = ",".join(args)
    body = "\n".join(f"  {b}" for b in body)

    # Compute the text of the entire function.
    txt = f" def {name}({args}){return_annotation}:\n{body}"

    local_vars = ", ".join(locals.keys())
    txt = f"def __create_fn__({local_vars}):\n{txt}\n return {name}"

    ns = {}
    exec(txt, globals, ns)  # noqa: W0122
    fn = ns["__create_fn__"](**locals)

    if doc is not None:
        fn.__doc__ = doc

    return fn


def get_class_globals(clz):
    if clz.__module__ in sys.modules:
        globals = sys.modules[clz.__module__].__dict__.copy()
    else:
        globals = {}

    return globals
