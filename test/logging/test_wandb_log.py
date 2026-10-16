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
@pytest.mark.parametrize(
    "item, expected",
    [
        pytest.param(
            {"Energy": jnp.array(1.0), "acceptance": 0.7},
            {"Energy": 1.0, "acceptance": 0.7},
            id="scalars",
        ),
        pytest.param(
            {"Energy": {"Mean": 1.5, "Variance": 0.1}},
            {"Energy/Mean": 1.5, "Energy/Variance": 0.1},
            id="nested",
        ),
        pytest.param(
            {"z": jnp.array(1.0 + 2.0j)},
            {"z/re": 1.0, "z/im": 2.0},
            id="complex",
        ),
    ],
)
def test_wandb_log_metrics(offline_wandb, item, expected):
    log = nk.logging.WandbLog(project="test_metrics")
    for i in range(3):
        log(i, item)
    del log

    namespaces = {k.split("/")[0] for k in expected}
    for i, logged in enumerate(offline_wandb.logged):
        assert {k: logged[k] for k in expected} == pytest.approx(expected)
        # every namespace is plotted against its own step
        assert {logged[f"{ns}/step"] for ns in namespaces} == {i}
    assert len(offline_wandb.logged) == 3


@common.skipif_distributed
def test_wandb_log_run_id(offline_wandb):
    log = nk.logging.WandbLog(
        project="test_resume",
        run_id="nk-test-run-id",
        metadata={"L": 20},
        ignore=["skip_me"],
    )
    log(0, {"loss": 1.0, "skip_me": 2.0})
    assert log._run.id == "nk-test-run-id"
    assert log._run.config["L"] == 20
    del log

    (logged,) = offline_wandb.logged
    assert "loss" in logged
    assert not any(k.startswith("skip_me") for k in logged)


@common.skipif_distributed
def test_wandb_log_disabled(offline_wandb):
    log = nk.logging.WandbLog(project="test_disabled", mode="disabled")
    log(0, {"Energy": 1.0})
    assert log._run.disabled
    del log


@common.skipif_distributed
@pytest.mark.parametrize("save_params", [True, False])
def test_wandb_log_save_params(vstate, offline_wandb, save_params):
    log = nk.logging.WandbLog(
        project="test_params", save_params=save_params, save_params_every=3
    )
    for i in range(6):
        log(i, {"Energy": 1.0}, vstate)
    run_id = log._run.id
    del log

    # Steps 0 and 3 trigger a save
    n_saved = 2 if save_params else 0
    assert offline_wandb.artifacts == [f"{run_id}-parameters"] * n_saved


@common.skipif_distributed
def test_wandb_log_as_callback(vstate, offline_wandb):
    hi = vstate.hilbert
    ha = nk.operator.Ising(hi, nk.graph.Chain(hi.size), h=1.0)
    driver = nk.driver.VMC(ha, nk.optimizer.Sgd(0.01), variational_state=vstate)

    log = nk.logging.WandbLog(
        project="test_callback", save_params_every=2, log_histograms=True
    )
    driver.run(3, out=(), callback=log, show_progress=False)
    run_id = log._run.id

    assert [d["Energy/step"] for d in offline_wandb.logged] == [0, 1, 2]
    assert "Energy/Mean" in offline_wandb.logged[0]
    # histograms are logged together with the parameters
    assert "histograms/Dense/kernel" in offline_wandb.logged[0]
    assert "histograms/Dense/kernel" not in offline_wandb.logged[1]
    # steps 0 and 2, the flush at the end of the run, and the final state
    assert offline_wandb.artifacts == [f"{run_id}-parameters"] * 3 + [
        f"{run_id}-final-state"
    ]
    del log


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
