import dataclasses
import abc


@dataclasses.dataclass(frozen=True)
class ModuleFramework(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def is_loaded() -> bool:
        pass

    @staticmethod
    @abc.abstractmethod
    def is_my_module(module):
        pass

    @staticmethod
    @abc.abstractmethod
    def wrap(clz, module):
        return module

    @staticmethod
    def wrap_params(variables):
        return freeze({"params": variables})

    @staticmethod
    @abc.abstractmethod
    def unwrap_params(wrapped_variables):
        return wrapped_module


registered_frameworks = []


def framework(clz):
    """
    Registers a framework and it's wrapper methods to make it
    behave like a flax framework.
    """
    clz = dataclasses.dataclass(frozen=True)(clz)
    registered_frameworks.append(clz)
    return clz


@dataclasses.dataclass(frozen=True)
class UnknownFramework(ModuleFramework):
    name: str = "Unknown"

    @staticmethod
    def is_loaded() -> bool:
        return True

    @staticmethod
    def is_my_module(module):
        return False

    @staticmethod
    def wrap(module):
        return module

    @staticmethod
    def unwrap_params(params):
        return params


def identify_framework(module):
    for _framework in registered_frameworks:
        if _framework.is_my_module(module):
            return _framework

    return UnknownFramework


def maybe_wrap_module(module):
    """
    Passing a module from an unknown framework (might be user defined module, a jax
    module, flax or haiku or anything else really), attempt to identify what is the
    package/framework it comes from, and if so it correctly wraps it in order to
    make it behave like a flax module (our default).

    Also returns a function used to unpack the parameters once we are done.
    """
    framewrk = identify_framework(module)

    return framewrk, framewrk.wrap(module)
