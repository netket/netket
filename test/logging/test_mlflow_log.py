import pytest

import jax
from jax.nn.initializers import normal
from jax import numpy as jnp

import netket as nk

from .. import common


@pytest.fixture()
def vstate():
    N = 8
    hi = nk.hilbert.Spin(1 / 2, N)
    ma = nk.models.RBM(
        alpha=1,
        param_dtype=float,
        hidden_bias_init=normal(),
        visible_bias_init=normal(),
    )
    return nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), ma)


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    yield mlflow
    try:
        mlflow.end_run()
    except Exception:
        pass


@common.skipif_distributed
def test_mlflow_log_basic(vstate, isolated_mlflow):
    mlflow = isolated_mlflow

    log = nk.logging.MLFlowLog(experiment_name="test_basic", run_name="run0")

    for i in range(10):
        log(i, {"Energy": jnp.array(1.0), "acceptance": jnp.array(0.7)}, vstate)

    log.flush()
    run_id = log._run.info.run_id
    del log

    client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(run_id, "Energy")
    assert len(history) == 10
    assert all(m.value == pytest.approx(1.0) for m in history)

    history_acc = client.get_metric_history(run_id, "acceptance")
    assert len(history_acc) == 10


@common.skipif_distributed
def test_mlflow_log_nested_keys(vstate, isolated_mlflow):
    mlflow = isolated_mlflow

    log = nk.logging.MLFlowLog(experiment_name="test_nested")

    # Stats-like nested dict
    item = {"Energy": {"Mean": 1.5, "Variance": 0.1, "Sigma": 0.05}}
    log(0, item, vstate)

    run_id = log._run.info.run_id
    del log

    client = mlflow.tracking.MlflowClient()
    assert len(client.get_metric_history(run_id, "Energy.Mean")) == 1
    assert len(client.get_metric_history(run_id, "Energy.Variance")) == 1
    assert len(client.get_metric_history(run_id, "Energy.Sigma")) == 1


@common.skipif_distributed
def test_mlflow_log_complex(vstate, isolated_mlflow):
    mlflow = isolated_mlflow

    log = nk.logging.MLFlowLog(experiment_name="test_complex")
    log(0, {"z": jnp.array(1.0 + 2.0j)}, vstate)

    run_id = log._run.info.run_id
    del log

    client = mlflow.tracking.MlflowClient()
    re_hist = client.get_metric_history(run_id, "z.re")
    im_hist = client.get_metric_history(run_id, "z.im")
    assert len(re_hist) == 1
    assert len(im_hist) == 1
    assert re_hist[0].value == pytest.approx(1.0)
    assert im_hist[0].value == pytest.approx(2.0)


@common.skipif_distributed
def test_mlflow_log_with_tags(isolated_mlflow):
    mlflow = isolated_mlflow

    tags = {"model": "RBM", "L": "20"}
    log = nk.logging.MLFlowLog(experiment_name="test_tags", tags=tags)
    log(0, {"Energy": 1.0})

    run_id = log._run.info.run_id
    del log

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    assert run.data.tags.get("model") == "RBM"
    assert run.data.tags.get("L") == "20"


@common.skipif_distributed
def test_mlflow_log_resume(isolated_mlflow):
    mlflow = isolated_mlflow

    # First run: log 5 steps
    log1 = nk.logging.MLFlowLog(experiment_name="test_resume", run_name="run_a")
    for i in range(5):
        log1(i, {"loss": float(i)})
    run_id = log1._run.info.run_id
    del log1

    # Resume: log 5 more steps
    log2 = nk.logging.MLFlowLog(experiment_name="test_resume", run_id=run_id)
    for i in range(5, 10):
        log2(i, {"loss": float(i)})
    del log2

    client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(run_id, "loss")
    assert len(history) == 10


@common.skipif_distributed
def test_lazy_init(isolated_mlflow):
    mlflow = isolated_mlflow

    log = nk.logging.MLFlowLog(experiment_name="test_lazy")
    # Do not call the logger — run should not have been created
    assert log._run is None

    # No active run was started
    assert mlflow.active_run() is None


@common.skipif_distributed
def test_mlflow_log_save_params(vstate, tmp_path, isolated_mlflow):
    mlflow = isolated_mlflow

    log = nk.logging.MLFlowLog(
        experiment_name="test_params",
        save_params=True,
        save_params_every=3,
    )

    for i in range(6):
        log(i, {"Energy": 1.0}, vstate)

    run_id = log._run.info.run_id
    del log

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, path="parameters")
    # Steps 0 and 3 should have triggered a save → 2 artifacts
    assert len(artifacts) >= 1


@common.skipif_distributed
def test_mlflow_log_no_save_params(vstate, isolated_mlflow):
    mlflow = isolated_mlflow

    log = nk.logging.MLFlowLog(experiment_name="test_no_params", save_params=False)

    for i in range(5):
        log(i, {"Energy": 1.0}, vstate)

    run_id = log._run.info.run_id
    del log

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, path="parameters")
    assert len(artifacts) == 0


@common.skipif_distributed
def test_repr(isolated_mlflow):
    log = nk.logging.MLFlowLog(experiment_name="test_repr", run_name="my_run")
    assert "MLFlowLog" in repr(log)
    # Before first call, run_id is None
    assert "None" in repr(log)

    log(0, {"x": 1.0})
    # After first call, run_id is populated
    assert "None" not in repr(log)
    del log


@common.skipif_distributed
def test_save_params_every_zero_raises(isolated_mlflow):
    with pytest.raises(ValueError, match="save_params_every"):
        nk.logging.MLFlowLog(save_params=True, save_params_every=0)


@common.skipif_distributed
def test_metadata_not_relogged_on_resume(isolated_mlflow):
    mlflow = isolated_mlflow

    # Simulate a first run where metadata was logged via on_run_start
    mlflow.set_experiment("test_meta_resume")
    with mlflow.start_run(run_name="run_meta") as run:
        mlflow.log_params({"lr": "0.01"})
        run_id = run.info.run_id

    # Resuming with run_id should not raise due to re-logging the same param
    log2 = nk.logging.MLFlowLog(
        run_id=run_id,
        metadata={"lr": 0.01},
    )
    log2(1, {"loss": 0.9})  # would raise MlflowException if params re-logged
    del log2


@common.onlyif_distributed
def test_write_only_on_master(vstate, isolated_mlflow):
    rank = jax.process_index()

    log = nk.logging.MLFlowLog(experiment_name="test_distributed")

    for i in range(5):
        log(i, {"Energy": jnp.array(1.0)}, vstate)

    if rank == 0:
        assert log._run is not None
    else:
        assert log._run is None

    del log
