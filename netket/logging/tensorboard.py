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

from numbers import Number

from netket.utils.optional_deps import import_optional_dependency

from .base import AbstractLog


def tree_log(tree, root, data):
    """
    Maps all elements in tree, recursively calling tree_log with a new root string,
    and when it reaches leaves pushes (string, leave) tuples to data.

    Args:
        tree: a pytree where the leaf nodes contain data
        root: the root of the tags used to log to tensorboard
        data: a container modified in place

    """
    if tree is None:
        return
    elif isinstance(tree, list):
        for i, val in enumerate(tree):
            tree_log(val, root + f"/{i}", data)

    elif isinstance(tree, list) and hasattr(tree, "_fields"):
        for key in tree._fields:
            tree_log(getattr(tree, key), root + f"/{key}", data)

    elif isinstance(tree, tuple):
        for i, val in enumerate(tree):
            tree_log(val, root + f"/{i}", data)

    elif isinstance(tree, dict):
        for key, value in tree.items():
            tree_log(value, root + f"/{key}", data)  # noqa: F722

    elif hasattr(tree, "to_compound"):
        tree_log(tree.to_compound()[1], root, data)  # noqa: F722

    elif hasattr(tree, "to_dict"):
        tree_log(tree.to_dict(), root, data)  # noqa: F722

    elif isinstance(tree, complex):
        tree_log(tree.real, root + "/re", data)  # noqa: F722
        tree_log(tree.imag, root + "/im", data)  # noqa: F722

    else:
        data.append((root, tree))


class TensorBoardLog(AbstractLog):
    """
    Creates a tensorboard logger using tensorboardX's summarywriter.

    Refer to its documentation for further details

    https://tensorboardx.readthedocs.io/en/latest/tensorboard.html

    TensorBoardX must be installed.

    Args:
        logdir (string): Save directory location. Default is
          runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run.
          Use hierarchical folder structure to compare
          between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
          for each new experiment to compare across them.
        comment (string): Comment logdir suffix appended to the default
          ``logdir``. If ``logdir`` is assigned, this argument has no effect.
        purge_step (int):
          When logging crashes at step :math:`T+X` and restarts at step :math:`T`,
          any events whose global_step larger or equal to :math:`T` will be
          purged and hidden from TensorBoard.
          Note that crashed and resumed experiments should have the same ``logdir``.
        max_queue (int): Size of the queue for pending events and
          summaries before one of the 'add' calls forces a flush to disk.
          Default is ten items.
        flush_secs (int): How often, in seconds, to flush the
          pending events and summaries to disk. Default is every two minutes.
        filename_suffix (string): Suffix added to all event filenames in
          the logdir directory. More details on filename construction in
          tensorboard.summary.writer.event_file_writer.EventFileWriter.
        write_to_disk (boolean):
          If pass `False`, TensorBoardLog will not write to disk.

    Examples:
        Logging optimisation to tensorboard.

        >>> import pytest; pytest.skip("skip automated test of this docstring")
        >>>
        >>> import netket as nk
        >>> # create a summary writer with automatically generated folder name.
        >>> writer = nk.logging.TensorBoardLog()
        >>> # folder location: runs/May04_22-14-54_s-MacBook-Pro.local/
        >>> # create a summary writer using the specified folder name.
        >>> writer = nk.logging.TensorBoardLog("my_experiment")
        >>> # folder location: my_experiment
        >>> # create a summary writer with comment appended.
        >>> writer = nk.logging.TensorBoardLog(comment="LR_0.1_BATCH_16")
        >>> # folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self._init_args = args
        """Store the args for the lazily initialized SummaryWriter's constructor."""
        self._init_kwargs = kwargs
        """Store the kwargs for the lazily initialized SummaryWriter's constructor."""

        self._writer = None
        """Lazily initialized summarywriter constructor"""

        self._old_step = 0

    def _init_tensorboard(self):
        tensorboardX = import_optional_dependency(
            "tensorboardX", descr="TensorBoardLog"
        )

        self._writer = tensorboardX.SummaryWriter(*self._init_args, **self._init_kwargs)

    def __call__(self, step, item, machine):
        if self._writer is None:
            self._init_tensorboard()

        data = []
        tree_log(item, "", data)

        for key, val in data:
            if isinstance(val, Number):
                self._writer.add_scalar(key[1:], val, step)

        self._writer.flush()
        self._old_step = step

    def __del__(self):
        self.flush()

    def _flush_log(self):
        if self._writer is not None:
            self._writer.flush()

    def _flush_params(self, _):
        return None

    def flush(self, variational_state=None):
        """
        Writes to file the content of this logger.

        :param machine: optionally also writes the parameters of the machine.
        """
        self._flush_log()

        if variational_state is not None:
            self._flush_params(variational_state)
