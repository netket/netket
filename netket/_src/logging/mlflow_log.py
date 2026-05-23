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

import os
import tempfile
from numbers import Number
from typing import Any, TYPE_CHECKING

from netket.utils.optional_deps import import_optional_dependency
from netket.utils.tree_walk import walk_tree_with_path
from netket.jax.sharding import extract_replicated

from netket.logging.base import AbstractLog

if TYPE_CHECKING:
    import mlflow
    from netket.vqs import VariationalState


def _visit_leaf(root, tree, *, data):
    if isinstance(tree, complex):
        data.append((root + "/re", tree.real))
        data.append((root + "/im", tree.imag))
    elif (
        hasattr(tree, "real")
        and hasattr(tree, "imag")
        and not isinstance(tree, (int, float))
    ):
        # numpy/jax complex scalars
        real, imag = float(tree.real), float(tree.imag)
        if imag != 0.0:
            data.append((root + "/re", real))
            data.append((root + "/im", imag))
        else:
            data.append((root, real))
    else:
        data.append((root, tree))


class MLFlowLog(AbstractLog):
    """
    Logger that streams metrics and optional model checkpoints to an
    `MLflow <https://mlflow.org>`_ tracking server.

    On the first call the logger lazily starts an MLflow run so that
    constructing the logger does not create a run if it is never used.

    The ``mlflow`` package must be installed (``pip install mlflow``).

    Nested metric keys (e.g. ``Energy/Mean``) are mapped to MLflow metric
    names using ``.`` as separator (``Energy.Mean``), which MLflow's UI
    renders in a collapsible tree.  Complex-valued scalars are split into
    separate ``<key>/re`` and ``<key>/im`` entries.

    Args:
        experiment_name: Name of the MLflow experiment.  If ``None`` the
            currently active experiment (or the MLflow default) is used.
        run_name: Human-readable label attached to the new run.
        run_id: If provided, resumes an existing run instead of starting a
            fresh one.  Takes precedence over ``run_name``.
        tags: Optional dict of string key/value tags to attach to the run.
        save_params: If ``True``, periodically serialize model parameters
            as an MLflow artifact (MessagePack binary, same format as
            :class:`~netket.logging.JsonLog`).
        save_params_every: Save parameters every this many optimisation
            steps.  Only relevant when ``save_params=True``.

    Examples:
        Log an optimisation run to the local MLflow store.

        >>> import pytest; pytest.skip("skip automated test of this docstring")
        >>>
        >>> import netket as nk
        >>> logger = nk.logging.MLFlowLog(
        ...     experiment_name="Ising1d",
        ...     run_name="RBM_alpha1",
        ...     tags={"model": "RBM", "L": "20"},
        ... )
        >>> gs.run(n_iter=500, out=logger)

        Resume a previous run.

        >>> import pytest; pytest.skip("skip automated test of this docstring")
        >>>
        >>> import netket as nk
        >>> logger = nk.logging.MLFlowLog(run_id="<existing-run-id>")
        >>> gs.run(n_iter=500, out=logger)
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
        save_params: bool = True,
        save_params_every: int = 50,
    ):
        mlflow = import_optional_dependency("mlflow", descr="MLFlowLog")
        self._mlflow: mlflow = mlflow

        self._experiment_name = experiment_name
        self._run_name = run_name
        self._run_id = run_id
        self._tags = tags
        self._save_params = save_params
        self._save_params_every = save_params_every

        self._run = None
        self._old_step = 0
        self._steps_since_last_params = 0

    def _init_mlflow(self):
        if self._experiment_name is not None:
            mlflow.set_experiment(self._experiment_name)

        self._run = mlflow.start_run(
            run_id=self._run_id,
            run_name=self._run_name,
            tags=self._tags,
        )
        self._mlflow = mlflow

    def __call__(
        self,
        step: int,
        item: dict[str, Any],
        variational_state: "VariationalState | None" = None,
    ):
        if not self._is_master_process:
            return

        if self._run is None:
            self._init_mlflow()

        data: list[tuple[str, Any]] = []
        walk_tree_with_path(item, "", visit_leaf=_visit_leaf, data=data)

        metrics = {}
        for key, val in data:
            mlflow_key = key.lstrip("/").replace("/", ".")
            if isinstance(val, Number) and not isinstance(val, bool):
                metrics[mlflow_key] = float(val)

        if metrics:
            self._mlflow.log_metrics(metrics, step=step)

        if (
            variational_state is not None
            and self._save_params
            and self._steps_since_last_params % self._save_params_every == 0
        ):
            self._flush_params(variational_state, step)

        self._steps_since_last_params += 1
        self._old_step = step

    def _flush_params(self, variational_state: "VariationalState", step: int):
        from flax import serialization

        binary_data = serialization.to_bytes(
            extract_replicated(variational_state.variables)
        )
        fd, fname = tempfile.mkstemp(suffix=f"_step{step}.mpack")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(binary_data)
            self._mlflow.log_artifact(fname, artifact_path="parameters")
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
        self.flush()
        if self._run is not None and self._mlflow is not None:
            self._mlflow.end_run()

    def __repr__(self):
        run_id = self._run.info.run_id if self._run is not None else None
        return (
            f"MLFlowLog(experiment={self._experiment_name!r}, "
            f"run_name={self._run_name!r}, run_id={run_id!r})"
        )
