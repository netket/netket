import sys
import builtins
import inspect

from dataclasses import MISSING


def get_own_class_annotations(cls):
    """Return annotations defined directly on cls (not inherited from parents).

    In Python < 3.14, class annotations are stored in ``cls.__dict__`` and can
    be inherited if the class defines none itself.  We read directly from
    ``__dict__`` so that we see only the annotations belonging to *this* class.

    In Python 3.14+ (PEP 649 / annotationlib), ``__annotations__`` is no longer
    stored inside ``__dict__``; instead it is always a *per-class* attribute that
    is never shared with parent classes.  Accessing ``cls.__annotations__``
    therefore already gives us the right (non-inherited) dict.
    """
    if "__annotations__" in cls.__dict__:
        # Python < 3.14: annotations live in __dict__ and are always own annotations
        return cls.__dict__["__annotations__"]
    elif sys.version_info >= (3, 14):
        # Python 3.14+: __annotations__ is per-class and not in __dict__
        return cls.__annotations__
    return {}


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


def maximum_positional_args(fun) -> int | float:
    """
    Given a function, returns an integer that represents the maximum number
    of position or infinity.
    """
    sig = inspect.signature(fun)
    parameters = sig.parameters.values()

    max_positional_args: int | float = 0
    for param in parameters:
        if param.kind in [
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ]:
            max_positional_args += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            # This means the function accepts an arbitrary number of positional arguments
            max_positional_args = float("inf")
            break
    return max_positional_args


def keyword_arg_names(fun) -> list[str]:
    """
    Given a function, returns a list of the argument names that can be passed to it
    with a keyword.
    """
    sig = inspect.signature(fun)
    parameters = sig.parameters.values()

    names = []
    for param in parameters:
        if param.kind in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ]:
            names.append(param.name)
    return names
