from typing import Any
from ..base import HilbertIndex

_constrained_indices: dict[Any, HilbertIndex] = {}


def register_constrained_hilbert_index(constraint, factory_fun):
    _constrained_indices[constraint] = factory_fun


def get_specialized_constrained_hilbert_index(constraint, local_states, size):
    factory_fun = _constrained_indices.get(constraint, None)
    if factory_fun is not None:
        return factory_fun(constraint, local_states, size)
