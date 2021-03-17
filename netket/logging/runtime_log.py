import dataclasses
from functools import partial

from os import path as _path

from flax import serialization

from jax.tree_util import tree_map

from netket.utils import History, accum_histories_in_tree


class RuntimeLog:
    """
    Runtim Logger, that can be passed with keyword argument `logger` to Monte
    Carlo drivers in order to serialize the outpit data of the simulation.

    This logger keeps the data in memory, and does not save it to disk.
    """

    def __init__(self):
        """
        Crates a Runtime Logger.
        """
        self._data = None
        self._old_step = 0

    def __call__(self, step, item, variational_state):
        self._data = accum_histories_in_tree(self._data, item, step=step)
        self._old_step = step

    @property
    def data(self):
        """
        The dictionary of logged data.
        """
        return self._data

    def __getitem__(self, key):
        return self.data[key]

    def flush(self, variational_state):
        pass
