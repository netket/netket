import json as _json
from os import path as _path
from netket.vmc_common import tree_map as _tree_map
from netket.stats.mc_stats import Stats

from tensorboardX import SummaryWriter


def tree_log(tree, root, data):
    """
    Maps all elements in tree, recursively calling tree_log with a new root string,
    and when it reaches leaves pushes (string, leave) tuples to data.
    """
    if tree is None:
        return data
    elif isinstance(tree, list):
        tmp = [
            tree_log(val, root + "/{}".format(i), data) for (i, val) in enumerate(tree)
        ]
        return data
    elif isinstance(tree, list) and hasattr(tree, "_fields"):
        tmp = [
            tree_log(getattr(tree, key), root + "/{}".format(key), data)
            for key in tree._fields
        ]
        return data
    elif isinstance(tree, tuple):
        tmp = tuple(
            tree_log(val, root + "/{}".format(i), data) for (i, val) in enumerate(tree)
        )
        return data
    elif isinstance(tree, dict):
        return {
            key: tree_log(value, root + "/{}".format(key), data)
            for key, value in tree.items()
        }
    else:
        data.append((root, tree))
        return data


class TBLog:
    """
    Creates a Json Logger sink object, that can be passed with keyword argument `logger` to Monte
    Carlo drivers in order to serialize the outpit data of the simulation.

    Args:
        output_prefix: the name of the output files before the extension
        save_params_every: every how many iterations should machine parameters be flushed to file
        write_every: every how many iterations should data be flushed to file
        mode: Specify the behaviour in case the file already exists at this output_prefix. Options
        are
        - `[w]rite`: (default) overwrites file if it already exists;
        - `[a]ppend`: appends to the file if it exists, overwise creates a new file;
        - `[x]` or `fail`: fails if file already exists;
    """

    def __init__(
        self, *args, save_params_every=50, **kwargs,
    ):

        self._writer = SummaryWriter(*args, **kwargs)

        self._save_params_every = save_params_every
        self._old_step = 0

    def __call__(self, step, item, machine):

        data = []
        tree_log(item, "", data)

        for key, val in data:
            if isinstance(val, Stats):
                val = val.mean

            if isinstance(val, complex):
                self._writer.add_scalar(key[1:] + "/re", val.real, step)
                self._writer.add_scalar(key[1:] + "/im", val.imag, step)
            else:
                self._writer.add_scalar(key[1:], val, step)

        self._writer.flush()
        self._old_step = step

    def _flush_log(self):
        self._writer.flush()

    def _flush_params(self, machine):
        return None

    def flush(self, machine=None):
        """
        Writes to file the content of this logger.

        :param machine: optionally also writes the parameters of the machine.
        """
        self._flush_log()

        if machine is not None:
            self._flush_params(machine)
