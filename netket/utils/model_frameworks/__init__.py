"""
This module attempts to autodetect when a model/Module passed to netket
comes from one of the several jax packages in existance.

It it comes from jax, flax, haiku or whatever else, and then extracts
the two functions that are really needed (init_fun and apply_fun).

If you want to add support for another framework, you should add
a new file in this folder and include it here.
"""


from .base import maybe_wrap_module, registered_frameworks

from . import flax, jax
