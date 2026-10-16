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
import os
import tempfile
from collections.abc import Collection
from numbers import Number
from typing import Any, TYPE_CHECKING

import jax
import numpy as np

from netket.utils.optional_deps import import_optional_dependency
from netket.utils.tree_walk import walk_tree_with_path
from netket.jax.sharding import extract_replicated
from netket._src.callbacks.base import AbstractCallback
from netket._src.logging.mlflow_log import _expand_node, _visit_leaf
from netket.utils import struct

if TYPE_CHECKING:
    from netket.vqs import VariationalState


class WandbLog(AbstractCallback):
    """
    Logger that streams metrics and optional model checkpoints to
    `Weights & Biases <https://wandb.ai>`_.

    On the first call the logger lazily starts a wandb run so that
    constructing the logger does not create a run if it is never used.

    The ``wandb`` package must be installed (``pip install wandb``).

    Nested metric keys (e.g. ``Energy/Mean``) are logged with ``/`` preserved,
    which wandb's UI renders as collapsible groups.  Complex-valued scalars are
    split into separate ``<key>/re`` and ``<key>/im`` entries.

    NetKet loggers can be called more than once with the same ``step`` (for
    different keys), while wandb expects a single, monotonically increasing
    global step.  For this reason, every top-level namespace (the first path
    segment of a key, e.g. ``Energy``) gets its own hidden ``<namespace>/step``
    metric, declared with :func:`wandb.define_metric`, and its metrics are
    plotted against it.

    This class is a full :class:`~netket.callbacks.AbstractCallback` and can be passed
    either as ``out=logger`` *or* as ``callback=logger``.  When
    used as a callback the logger automatically captures the variational state snapshot
    taken just before the parameter update.

    Args:
        project: Name of the wandb project.  If ``None`` wandb uses its own
            default (the ``WANDB_PROJECT`` environment variable, or
            ``"uncategorized"``).
        entity: wandb entity (user or team) that owns the run.
        run_name: Human-readable label attached to the new run.
        run_id: If provided, the run with this id is resumed if it exists, or
            created with this id otherwise.  Useful to resume an interrupted
            job in place.
        group: Group name used to cluster related runs in the wandb UI.
        job_type: Free-form job type of the run (e.g. ``"vmc"``), useful to
            filter the runs of a group.
        tags: Optional collection of string tags to attach to the run.
        mode: wandb mode, one of ``"online"``, ``"offline"`` or
            ``"disabled"``.  If ``None``, wandb's default is used (which can be
            set with the ``WANDB_MODE`` environment variable).
        save_params: If ``True``, periodically serialize model parameters
            as a versioned wandb artifact (MessagePack binary, same format as
            :class:`~netket.logging.JsonLog`).
        save_params_every: Save parameters every this many optimisation
            steps.  Only relevant when ``save_params=True``.
        save_final_state: If ``True``, the variational state at the end of
            the run is uploaded as a separate artifact.
        log_histograms: If ``True``, log a histogram of every parameter
            every time the parameters are saved.
        metadata: Optional dict of hyper-parameters stored in the wandb
            ``config`` of the run.
        ignore: Optional collection of top-level log-dict keys to skip
            entirely.  Useful for suppressing entries that cannot be
            serialised as wandb metrics (e.g. large arrays or custom
            objects).

    .. tip::
        Use ``metadata`` to attach a dict of hyper-parameters (learning rate,
        system size, model type, …) to the run.  They are stored in the wandb
        *config* of the run, making it easy to filter and compare runs in the
        wandb UI without external bookkeeping.

    Examples:
        Log an optimisation run to wandb.

        >>> import pytest; pytest.skip("skip automated test of this docstring")
        >>>
        >>> import netket as nk
        >>> logger = nk.logging.WandbLog(
        ...     project="Ising1d",
        ...     run_name="RBM_alpha1",
        ...     tags=["RBM"],
        ...     metadata={"learning_rate": 0.01, "alpha": 1, "L": 20},
        ... )
        >>> gs.run(n_iter=500, out=logger)

        Log offline, to synchronise later with ``wandb sync``.

        >>> import pytest; pytest.skip("skip automated test of this docstring")
        >>>
        >>> import netket as nk
        >>> logger = nk.logging.WandbLog(project="Ising1d", mode="offline")
        >>> gs.run(n_iter=500, out=logger)

        Resume a previous run.

        >>> import pytest; pytest.skip("skip automated test of this docstring")
        >>>
        >>> import netket as nk
        >>> logger = nk.logging.WandbLog(project="Ising1d", run_id="<run-id>")
        >>> gs.run(n_iter=500, out=logger)

        Using the logger as a callback.

        >>> import pytest; pytest.skip("skip automated test of this docstring")
        >>>
        >>> import netket as nk
        >>> logger = nk.logging.WandbLog(project="Ising1d")
        >>> gs.run(n_iter=500, callback=logger)
    """

    _metadata: Any = struct.field(pytree_node=False, serialize=False)
    _callback_pending_vstate: Any = struct.field(pytree_node=False, serialize=False)
    _wandb: Any = struct.field(pytree_node=False, serialize=False)
    _project: Any = struct.field(pytree_node=False, serialize=False)
    _entity: Any = struct.field(pytree_node=False, serialize=False)
    _run_name: Any = struct.field(pytree_node=False, serialize=False)
    _run_id: Any = struct.field(pytree_node=False, serialize=False)
    _group: Any = struct.field(pytree_node=False, serialize=False)
    _job_type: Any = struct.field(pytree_node=False, serialize=False)
    _tags: Any = struct.field(pytree_node=False, serialize=False)
    _mode: Any = struct.field(pytree_node=False, serialize=False)
    _save_params: Any = struct.field(pytree_node=False, serialize=False)
    _save_params_every: Any = struct.field(pytree_node=False, serialize=False)
    _save_final_state: Any = struct.field(pytree_node=False, serialize=False)
    _log_histograms: Any = struct.field(pytree_node=False, serialize=False)
    _ignore: Any = struct.field(pytree_node=False, serialize=False)
    _run: Any = struct.field(pytree_node=False, serialize=False)
    _defined_metrics: Any = struct.field(pytree_node=False, serialize=False)
    _old_step: Any = struct.field(pytree_node=False, serialize=False)
    _steps_since_last_params: Any = struct.field(pytree_node=False, serialize=False)

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
        run_id: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        tags: Collection[str] | None = None,
        mode: str | None = None,
        save_params: bool = True,
        save_params_every: int = 50,
        save_final_state: bool = True,
        log_histograms: bool = False,
        metadata: dict | None = None,
        ignore: Collection[str] | None = None,
    ):
        super().__init__()
        self._metadata = metadata or {}
        self._callback_pending_vstate = None

        wandb = import_optional_dependency("wandb", descr="WandbLog")
        self._wandb: wandb = wandb

        self._project = project
        self._entity = entity
        self._run_name = run_name
        self._run_id = run_id
        self._group = group
        self._job_type = job_type
        self._tags = list(tags) if tags is not None else None
        self._mode = mode
        self._save_params = save_params
        self._save_params_every = save_params_every
        self._save_final_state = save_final_state
        self._log_histograms = log_histograms
        self._ignore = frozenset(ignore) if ignore is not None else frozenset()

        if save_params and save_params_every < 1:
            raise ValueError("`save_params_every` must be >= 1")

        self._run = None
        self._defined_metrics = set()
        self._old_step = 0
        self._steps_since_last_params = 0

    def _init_wandb(self):
        self._run = self._wandb.init(
            project=self._project,
            entity=self._entity,
            name=self._run_name,
            id=self._run_id,
            group=self._group,
            job_type=self._job_type,
            tags=self._tags,
            config=dict(self._metadata) or None,
            mode=self._mode,
            resume="allow" if self._run_id is not None else None,
        )

    def _prepare_metric(self, key: str, step: int, log: dict[str, Any]):
        # Plot every metric against the step of its top-level namespace.
        namespace = key.split("/", 1)[0]
        step_key = f"{namespace}/step"
        if step_key not in self._defined_metrics:
            self._run.define_metric(step_key, hidden=True)
            self._defined_metrics.add(step_key)
        if key not in self._defined_metrics:
            self._run.define_metric(key, step_metric=step_key)
            self._defined_metrics.add(key)
        log[step_key] = step

    def __call__(
        self,
        step: int,
        item: dict[str, Any],
        variational_state: "VariationalState | None" = None,
    ):
        # Materialise the log tree on all ranks before i/o to avoid deadlocks:
        # walking the item can trigger a host-side collective (reducing a sharded
        # Stats, materialising a sharded array), which must run on every rank.
        data: list[tuple[str, Any]] = []
        if self._ignore:
            item = {k: v for k, v in item.items() if k not in self._ignore}
        walk_tree_with_path(
            item, "", visit_leaf=_visit_leaf, expand_node=_expand_node, data=data
        )

        if not self._is_master_process:
            return

        if self._run is None:
            self._init_wandb()

        log: dict[str, Any] = {}
        for key, val in data:
            if isinstance(val, Number) and not isinstance(val, bool):
                wandb_key = key.lstrip("/")
                self._prepare_metric(wandb_key, step, log)
                log[wandb_key] = float(val)

        if (
            variational_state is not None
            and self._save_params
            and self._steps_since_last_params % self._save_params_every == 0
        ):
            self._flush_params(variational_state, step)
            if self._log_histograms:
                self._add_histograms(variational_state, step, log)

        if log:
            self._run.log(log)

        self._steps_since_last_params += 1
        self._old_step = step

    def _add_histograms(
        self, variational_state: "VariationalState", step: int, log: dict[str, Any]
    ):
        params = extract_replicated(variational_state.parameters)
        for path, leaf in jax.tree_util.tree_leaves_with_path(params):
            name = "histograms/" + "/".join(
                str(getattr(p, "key", getattr(p, "idx", p))) for p in path
            )
            values = np.asarray(leaf)
            if np.iscomplexobj(values):
                components = {f"{name}/re": values.real, f"{name}/im": values.imag}
            else:
                components = {name: values}
            for key, comp in components.items():
                self._prepare_metric(key, step, log)
                log[key] = self._wandb.Histogram(comp.ravel())

    def _flush_params(
        self, variational_state: "VariationalState", step: int, *, final: bool = False
    ):
        from flax import serialization

        binary_data = serialization.to_bytes(
            extract_replicated(variational_state.variables)
        )
        kind = "final-state" if final else "parameters"
        fd, fname = tempfile.mkstemp(suffix=f"_step{step}.mpack")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(binary_data)
            artifact = self._wandb.Artifact(name=f"{self._run.id}-{kind}", type="model")
            artifact.add_file(fname, name=f"{kind}.mpack")
            self._run.log_artifact(artifact)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def flush(self, variational_state: "VariationalState | None" = None):
        """
        Flushes pending data and optionally saves model parameters.

        Args:
            variational_state: if provided and ``save_params=True``, the
                current model parameters are uploaded as an artifact.
        """
        if not self._is_master_process:
            return

        if variational_state is not None and self._save_params:
            if self._run is not None:
                self._flush_params(variational_state, self._old_step)

    def __del__(self):
        # __init__ might have failed before setting _run (e.g. missing wandb)
        if getattr(self, "_run", None) is not None:
            self._run.finish()

    def __repr__(self):
        run_id = self._run.id if self._run is not None else None
        return (
            f"WandbLog(project={self._project!r}, "
            f"run_name={self._run_name!r}, run_id={run_id!r})"
        )

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
        if self._run is None:
            self._init_wandb()

    def before_parameter_update(self, step, log_data, driver):
        self._callback_pending_vstate = copy.copy(driver.state)

    def on_step_end(self, step, log_data, driver):
        self(step, log_data, self._callback_pending_vstate)
        self._callback_pending_vstate = None

    def on_run_end(self, step, driver):
        self._flush_final(driver.state)
        self._callback_pending_vstate = None

    def on_run_error(self, step, error, driver):
        self._flush_final(driver.state)
        self._callback_pending_vstate = None

    def _flush_final(self, variational_state: "VariationalState"):
        self.flush(variational_state)
        if self._is_master_process and self._save_final_state and self._run is not None:
            self._flush_params(variational_state, self._old_step, final=True)
