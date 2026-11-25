import io
import logging
import sys

import numpy as np
import pytest

import netket as nk
from netket._src.progress import progress_bar


class _Tty(io.StringIO):
    def isatty(self):
        return True


class DummyDriver:
    _loss_name = "Energy"


@pytest.fixture
def setup(monkeypatch, caplog):
    """Set the log interval and stderr; INFO records are captured."""
    caplog.set_level(logging.INFO)

    def set_config(interval=60, terminal=False):
        monkeypatch.setattr(nk._src.progress, "_INTERVAL", interval)
        stderr = _Tty() if terminal else io.StringIO()
        monkeypatch.setattr(sys, "stderr", stderr)
        return stderr

    return set_config


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "netket.progress"]


@pytest.mark.parametrize("terminal", [True, False])
def test_draws_on_a_terminal_and_logs_otherwise(setup, caplog, terminal):
    stderr = setup(terminal=terminal)
    with progress_bar(total=3, desc="VMC") as pbar:
        pbar.update(3)
    assert ("VMC: 100%" in stderr.getvalue()) == terminal
    assert (_lines(caplog) != []) == (not terminal)


@pytest.mark.parametrize("dropped_by", ["logger level", "handler level"])
def test_auto_draws_when_progress_records_would_be_dropped(
    setup, caplog, monkeypatch, dropped_by
):
    # E.g. a plain script without logging configured would otherwise show nothing.
    stderr = setup()
    if dropped_by == "logger level":
        caplog.set_level(logging.WARNING)
    else:
        progress_logger = logging.getLogger("netket.progress")
        monkeypatch.setattr(progress_logger, "propagate", False)
        monkeypatch.setattr(progress_logger, "handlers", [logging.NullHandler()])
        progress_logger.handlers[0].setLevel(logging.WARNING)
    with progress_bar(total=3, desc="VMC") as pbar:
        pbar.update(3)
    assert "VMC: 100%" in stderr.getvalue()


@pytest.mark.parametrize(("interval", "n_lines"), [(0, 5), (3600, 1)])
def test_log_lines_are_throttled_and_end_on_the_final_state(
    setup, caplog, interval, n_lines
):
    stderr = setup(interval=interval)
    with progress_bar(total=5, desc="VMC") as pbar:
        for i in range(5):
            pbar.set_postfix_str(f"Energy={-i}", refresh=False)
            pbar.update(1)
    assert stderr.getvalue() == ""
    lines = _lines(caplog)
    # One line per update when the interval is 0 (the final state is not
    # repeated on close), else only the final one.
    assert len(lines) == n_lines, lines
    assert lines[-1].startswith("VMC: 100%|")
    assert "| 5/5 [" in lines[-1]
    assert lines[-1].endswith(", Energy=-4]")


def test_disabled_progress_shows_nothing(setup, caplog):
    stderr = setup(interval=0)
    with progress_bar(total=2, disable=True) as pbar:
        pbar.update(2)
    assert stderr.getvalue() == ""
    assert _lines(caplog) == []


def test_counter_assigned_as_numpy_scalar(setup, caplog):
    # The TDVP driver assigns `n` and calls refresh().
    setup(interval=0)
    with progress_bar(total=np.asarray(2.0), unit_scale=True) as pbar:
        pbar.n = np.asarray(1.5)
        pbar.refresh()
    assert "1.50/2.00" in _lines(caplog)[0]


def test_progress_bar_callback_logs_off_a_terminal(setup, caplog):
    setup(interval=3600)
    driver = DummyDriver()
    driver.step_count = 0
    driver._start_step = 0
    driver._loss_stats = "-1.5 ± 0.1"

    pb = nk._src.callbacks.progressbar.ProgressBarCallback(10)
    pb.on_run_start(0, driver)
    for step in range(10):
        pb.before_parameter_update(step, {}, driver)
        driver.step_count = step + 1
        pb.on_step_end(step, {}, driver)
    pb.on_run_end(driver.step_count, driver)

    lines = _lines(caplog)
    assert len(lines) == 1, lines
    assert "| 10/10 [" in lines[0]
    assert lines[0].endswith("Energy=-1.5 ± 0.1]")
