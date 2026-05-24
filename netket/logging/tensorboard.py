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

import copy
import json
from typing import Any, TYPE_CHECKING
from numbers import Number

import jax

from netket.utils.optional_deps import import_optional_dependency
from netket.utils.tree_walk import walk_tree_with_path
from netket.vqs import VariationalState
from netket._src.callbacks.base import AbstractCallback
from netket.utils import struct

if TYPE_CHECKING:
    import tensorboardX


def _collect_tensorboard_leaf(root, tree, data):
    data.append((root, tree))


def _write_tensorboard_leaf(root, tree, *, writer, step):
    if isinstance(tree, Number):
        writer.add_scalar(root.removeprefix("/"), tree, step)


def _expand_tensorboard_node(_root, tree, **_kwargs):
    if isinstance(tree, complex):
        return (("re", tree.real), ("im", tree.imag))
    return None


class TensorBoardLog(AbstractCallback):
    """
    Creates a tensorboard logger using tensorboardX's summarywriter.

    Refer to its documentation for further details

    https://tensorboardx.readthedocs.io/en/latest/tensorboard.html

    TensorBoardX must be installed.

    This class is a full :class:`~netket.callbacks.AbstractCallback` and can be passed
    either as ``out=logger`` *or* inside the ``callbacks=[..., logger]`` list.  When
    used as a callback the logger automatically captures the variational state snapshot
    taken just before the parameter update.

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
        metadata: Optional flat dict of key/value pairs written as a JSON text
            summary at the start of the run (visible in the TensorBoard Text tab).

    .. tip::
        Use ``metadata`` to attach a flat dict of hyper-parameters (learning rate,
        system size, model type, …) to the run.  They are written once as a JSON
        text entry and appear in the TensorBoard *Text* tab, keeping all run
        information in one place without external bookkeeping.

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

        Attaching metadata to record hyper-parameters.

        >>> import pytest; pytest.skip("skip automated test of this docstring")
        >>>
        >>> import netket as nk
        >>> writer = nk.logging.TensorBoardLog(
        ...     "my_experiment",
        ...     metadata={"learning_rate": 0.01, "alpha": 1, "L": 20},
        ... )
        >>> gs.run(n_iter=300, out=writer)
        >>> # hyper-parameters appear in the TensorBoard Text tab as JSON

        Using the logger as a callback.

        >>> import pytest; pytest.skip("skip automated test of this docstring")
        >>>
        >>> import netket as nk
        >>> writer = nk.logging.TensorBoardLog("my_experiment")
        >>> gs.run(n_iter=300, callbacks=[writer])
    """

    _metadata: Any = struct.field(pytree_node=False, serialize=False)
    _callback_pending_vstate: Any = struct.field(pytree_node=False, serialize=False)
    _init_args: Any = struct.field(pytree_node=False, serialize=False)
    _init_kwargs: Any = struct.field(pytree_node=False, serialize=False)
    _writer: Any = struct.field(pytree_node=False, serialize=False)
    _old_step: Any = struct.field(pytree_node=False, serialize=False)

    def __init__(
        self,
        *args,
        metadata: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self._metadata = metadata or {}
        self._callback_pending_vstate = None
        self._init_args = args
        self._init_kwargs = kwargs
        self._writer: tensorboardX.SummaryWriter | None = None
        self._old_step = 0

    def _init_tensorboard(self):
        tensorboardX = import_optional_dependency(
            "tensorboardX", descr="TensorBoardLog"
        )
        self._writer = tensorboardX.SummaryWriter(*self._init_args, **self._init_kwargs)

    def __call__(
        self,
        step: int,
        item: dict[str, Any],
        variational_state: VariationalState | None = None,
    ):
        if not self._is_master_process:
            return
        if self._writer is None:
            self._init_tensorboard()
        assert self._writer is not None

        walk_tree_with_path(
            item,
            "",
            visit_leaf=_write_tensorboard_leaf,
            expand_node=_expand_tensorboard_node,
            writer=self._writer,
            step=step,
        )

        self._writer.flush()
        self._old_step = step

    def __del__(self):
        self.flush()

    def flush(self, variational_state=None):
        """
        Writes to file the content of this logger.

        :param machine: optionally also writes the parameters of the machine.
        """
        if self._writer is not None:
            self._writer.flush()

    # --- AbstractLog compatibility ---

    @property
    def _is_master_process(self) -> bool:
        return jax.process_index() == 0

    # --- AbstractCallback interface ---

    @property
    def callback_order(self) -> int:
        return 10

    def on_run_start(self, step, driver):
        if not self._is_master_process:
            return
        if self._writer is None:
            self._init_tensorboard()
        if self._metadata:
            text = json.dumps(
                {str(k): str(v) for k, v in self._metadata.items()}, indent=2
            )
            self._writer.add_text("metadata", f"```json\n{text}\n```", global_step=0)

    def before_parameter_update(self, step, log_data, driver):
        self._callback_pending_vstate = copy.copy(driver.state)

    def on_step_end(self, step, log_data, driver):
        self(step, log_data, self._callback_pending_vstate)
        self._callback_pending_vstate = None

    def on_run_end(self, step, driver):
        self.flush(driver.state)
        self._callback_pending_vstate = None

    def on_run_error(self, step, error, driver):
        self.flush(driver.state)
        self._callback_pending_vstate = None
