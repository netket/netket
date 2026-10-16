import sys

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


@pytest.fixture()
def offline_wandb(tmp_path, monkeypatch):
    """
    Runs wandb offline in a temporary directory, and records what is sent to
    the run in `logged` (one dict per call to `run.log`) and `artifacts`.
    """
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")

    logged = []
    artifacts = []
    run_cls = wandb.sdk.wandb_run.Run
    orig_log = run_cls.log
    orig_log_artifact = run_cls.log_artifact

    def log(self, data, *args, **kwargs):
        logged.append(dict(data))
        return orig_log(self, data, *args, **kwargs)

    def log_artifact(self, artifact, *args, **kwargs):
        artifacts.append(artifact.name)
        return orig_log_artifact(self, artifact, *args, **kwargs)

    monkeypatch.setattr(run_cls, "log", log)
    monkeypatch.setattr(run_cls, "log_artifact", log_artifact)

    wandb.logged = logged
    wandb.artifacts = artifacts
    yield wandb
    wandb.finish()
    del wandb.logged, wandb.artifacts


@common.skipif_distributed
def test_wandb_log_basic(vstate, offline_wandb):
    log = nk.logging.WandbLog(project="test_basic", run_name="run0")

    for i in range(10):
        log(i, {"Energy": jnp.array(1.0), "acceptance": jnp.array(0.7)}, vstate)

    assert [d["Energy"] for d in offline_wandb.logged] == pytest.approx([1.0] * 10)
    assert [d["acceptance"] for d in offline_wandb.logged] == pytest.approx([0.7] * 10)
    # every namespace is plotted against its own step
    assert [d["Energy/step"] for d in offline_wandb.logged] == list(range(10))
    assert [d["acceptance/step"] for d in offline_wandb.logged] == list(range(10))
    assert log._run.name == "run0"
    del log


@common.skipif_distributed
def test_wandb_log_nested_keys(vstate, offline_wandb):
    log = nk.logging.WandbLog(project="test_nested")

    # Stats-like nested dict
    item = {"Energy": {"Mean": 1.5, "Variance": 0.1, "Sigma": 0.05}}
    log(0, item, vstate)
    del log

    (logged,) = offline_wandb.logged
    assert logged["Energy/Mean"] == pytest.approx(1.5)
    assert logged["Energy/Variance"] == pytest.approx(0.1)
    assert logged["Energy/Sigma"] == pytest.approx(0.05)
    assert logged["Energy/step"] == 0


@common.skipif_distributed
def test_wandb_log_complex(vstate, offline_wandb):
    log = nk.logging.WandbLog(project="test_complex")
    log(0, {"z": jnp.array(1.0 + 2.0j)}, vstate)
    del log

    (logged,) = offline_wandb.logged
    assert logged["z/re"] == pytest.approx(1.0)
    assert logged["z/im"] == pytest.approx(2.0)


@common.skipif_distributed
def test_wandb_log_tags_and_metadata(offline_wandb):
    log = nk.logging.WandbLog(
        project="test_tags",
        tags=["RBM", "L20"],
        group="scan",
        job_type="vmc",
        metadata={"learning_rate": 0.01, "L": 20},
    )
    log(0, {"Energy": 1.0})

    run = log._run
    assert set(run.tags) == {"RBM", "L20"}
    assert run.group == "scan"
    assert run.job_type == "vmc"
    assert run.config["learning_rate"] == pytest.approx(0.01)
    assert run.config["L"] == 20
    del log


@common.skipif_distributed
def test_wandb_log_run_id(offline_wandb):
    log = nk.logging.WandbLog(project="test_resume", run_id="nk-test-run-id")
    log(0, {"loss": 1.0})
    assert log._run.id == "nk-test-run-id"
    del log


@common.skipif_distributed
def test_lazy_init(offline_wandb):
    log = nk.logging.WandbLog(project="test_lazy")
    # Do not call the logger — run should not have been created
    assert log._run is None

    # No active run was started
    assert offline_wandb.run is None


@common.skipif_distributed
def test_wandb_log_disabled(offline_wandb):
    log = nk.logging.WandbLog(project="test_disabled", mode="disabled")
    log(0, {"Energy": 1.0})
    assert log._run.disabled
    del log


@common.skipif_distributed
def test_wandb_log_save_params(vstate, offline_wandb):
    log = nk.logging.WandbLog(
        project="test_params",
        save_params=True,
        save_params_every=3,
    )

    for i in range(6):
        log(i, {"Energy": 1.0}, vstate)
    run_id = log._run.id
    del log

    # Steps 0 and 3 should have triggered a save
    assert offline_wandb.artifacts == [f"{run_id}-parameters"] * 2


@common.skipif_distributed
def test_wandb_log_no_save_params(vstate, offline_wandb):
    log = nk.logging.WandbLog(project="test_no_params", save_params=False)

    for i in range(5):
        log(i, {"Energy": 1.0}, vstate)
    del log

    assert offline_wandb.artifacts == []


@common.skipif_distributed
def test_wandb_log_histograms(vstate, offline_wandb):
    log = nk.logging.WandbLog(
        project="test_histograms", log_histograms=True, save_params_every=2
    )

    for i in range(3):
        log(i, {"Energy": 1.0}, vstate)
    del log

    hist_keys = {
        "histograms/Dense/kernel",
        "histograms/Dense/bias",
        "histograms/visible_bias",
    }
    logged = offline_wandb.logged
    assert hist_keys <= set(logged[0])
    assert hist_keys.isdisjoint(logged[1])
    assert hist_keys <= set(logged[2])
    assert isinstance(logged[0]["histograms/Dense/kernel"], offline_wandb.Histogram)
    assert logged[2]["histograms/step"] == 2


@common.skipif_distributed
def test_wandb_log_as_callback(vstate, offline_wandb):
    hi = vstate.hilbert
    ha = nk.operator.Ising(hi, nk.graph.Chain(hi.size), h=1.0)
    driver = nk.driver.VMC(ha, nk.optimizer.Sgd(0.01), variational_state=vstate)

    log = nk.logging.WandbLog(project="test_callback", save_params_every=2)
    driver.run(3, out=(), callback=log, show_progress=False)
    run_id = log._run.id

    assert len(offline_wandb.logged) == 3
    assert [d["Energy/step"] for d in offline_wandb.logged] == [0, 1, 2]
    assert "Energy/Mean" in offline_wandb.logged[0]
    # steps 0 and 2, the flush at the end of the run, and the final state
    assert offline_wandb.artifacts == [f"{run_id}-parameters"] * 3 + [
        f"{run_id}-final-state"
    ]
    del log


@common.skipif_distributed
def test_repr(offline_wandb):
    log = nk.logging.WandbLog(project="test_repr", run_name="my_run")
    assert "WandbLog" in repr(log)
    # Before first call, run_id is None
    assert "run_id=None" in repr(log)

    log(0, {"x": 1.0})
    # After first call, run_id is populated
    assert "run_id=None" not in repr(log)
    del log


@common.skipif_distributed
def test_save_params_every_zero_raises(offline_wandb):
    with pytest.raises(ValueError, match="save_params_every"):
        nk.logging.WandbLog(save_params=True, save_params_every=0)


@common.skipif_distributed
def test_wandb_log_1d_array_expanded(offline_wandb):
    log = nk.logging.WandbLog(project="test_1d_array")
    log(0, {"vals": jnp.array([1.0, 2.0, 3.0])})
    del log

    (logged,) = offline_wandb.logged
    assert logged["vals/0"] == pytest.approx(1.0)
    assert logged["vals/1"] == pytest.approx(2.0)
    assert logged["vals/2"] == pytest.approx(3.0)


@common.skipif_distributed
def test_wandb_log_ignore(offline_wandb):
    log = nk.logging.WandbLog(project="test_ignore", ignore=["skip_me"])
    log(0, {"Energy": 1.0, "skip_me": jnp.array([1.0, 2.0, 3.0])})
    del log

    (logged,) = offline_wandb.logged
    assert "Energy" in logged
    # ignored key should produce no metrics at all
    assert not any(k.startswith("skip_me") for k in logged)


def test_missing_wandb_raises(monkeypatch):
    # Setting a module to None in sys.modules makes importing it fail.
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ModuleNotFoundError, match="pip install 'wandb'"):
        nk.logging.WandbLog()


@common.onlyif_distributed
def test_write_only_on_master(vstate, offline_wandb):
    rank = jax.process_index()

    log = nk.logging.WandbLog(project="test_distributed")

    for i in range(5):
        log(i, {"Energy": jnp.array(1.0)}, vstate)

    if rank == 0:
        assert log._run is not None
    else:
        assert log._run is None
        assert offline_wandb.logged == []

    del log
