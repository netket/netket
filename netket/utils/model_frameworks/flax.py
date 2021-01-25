from .base import ModuleFramework, framework


@framework
class FlaxFramework(ModuleFramework):

    name: str = "Flax"

    @staticmethod
    def is_loaded() -> bool:
        # this should be not necessary, as netket requires and loads
        # Flax, but let's set a good example
        return "flax" in sys.modules

    @staticmethod
    def is_my_module(module) -> bool:
        # this will only get callede if the module is loaded
        from flax import linen as nn

        return isinstance(module, nn.Module)

    @staticmethod
    def wrap(module):
        return module

    @staticmethod
    def wrap_params(variables):
        return variables

    @staticmethod
    def unwrap_params(params):
        return params
