"""Progress display for NetKet's long-running loops.

Loops get their progress object from :func:`progress_bar` and use only the
:class:`Progress` interface, so the backend can change without touching them.
On a terminal or in a notebook the backend is a tqdm bar. When stderr is not a
terminal (e.g. a batch job, where every bar repaint would be one more untimed
line in a file) and ``netket.progress`` INFO records are handled, it is
:class:`_LogProgress`, which writes the text the bar would show (description,
``n/total``, elapsed and remaining time, rate, postfix) to the
``netket.progress`` logger at ``INFO``, at most every ``_INTERVAL`` seconds and
once when it closes.
"""

import logging
import sys
import time
from numbers import Number
from typing import Protocol

from tqdm.auto import notebook_tqdm, std_tqdm, tqdm
from tqdm.std import EMA

logger = logging.getLogger("netket.progress")

# Seconds between two logged progress lines.
_INTERVAL = 60


class Progress(Protocol):
    """What a loop may do with the object returned by :func:`progress_bar`.

    Counts (``n``, ``update``) must be host numbers (Python or NumPy), not JAX
    arrays: a backend formats them on one process only.
    """

    n: float

    def update(self, n: float = 1) -> None: ...
    def set_postfix(
        self, ordered_dict=None, refresh: bool = True, **kwargs
    ) -> None: ...
    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None: ...
    def refresh(self) -> None:
        """Show the current state (``n`` may have been assigned directly)."""

    def unpause(self) -> None:
        """Leave the time since the last display out of the elapsed time."""

    def write(self, s: str) -> None:
        """Show a message; not silenced by ``disable``, so gate it by rank."""

    def close(self) -> None: ...
    def __enter__(self) -> "Progress": ...
    def __exit__(self, *exc) -> None: ...


def progress_bar(
    *,
    total: float | None = None,
    desc: str | None = None,
    unit: str = "it",
    initial: float = 0,
    unit_scale: bool = False,
    leave: bool = True,
    disable: bool = False,
    bar_format: str | None = None,
) -> Progress:
    """The progress display for a loop of ``total`` steps (see the module
    docstring for the backend). The arguments mean what they mean for
    :class:`tqdm.tqdm`; ``bar_format`` uses tqdm's format syntax."""
    kwargs = dict(
        total=total,
        desc=desc,
        unit=unit,
        initial=initial,
        unit_scale=unit_scale,
        bar_format=bar_format,
    )
    # A disabled display (e.g. on non-root processes) is a no-op tqdm.
    if disable or not _logs():
        return tqdm(leave=leave, disable=disable, dynamic_ncols=True, **kwargs)
    return _LogProgress(**kwargs)


def _logs() -> bool:
    """Whether progress is logged rather than drawn as a bar."""
    return (
        not sys.stderr.isatty()
        and notebook_tqdm is std_tqdm  # not a notebook
        and _info_is_handled()
    )


def _info_is_handled() -> bool:
    """Whether an INFO record of the ``netket.progress`` logger reaches a handler."""
    if not logger.isEnabledFor(logging.INFO):
        return False
    current = logger
    while current:
        if any(h.level <= logging.INFO for h in current.handlers):
            return True
        if not current.propagate:
            return False
        current = current.parent
    return False


class _LogProgress:
    """:class:`Progress` through the ``netket.progress`` logger.

    Updates, postfix changes and refreshes log the current state once the
    interval has passed since the last line; :meth:`close` logs the final state
    unless it was the last line. The rate is tqdm's (smoothed counts over
    smoothed durations), here between lines.
    """

    def __init__(self, *, total, desc, unit, initial, unit_scale, bar_format):
        self.n = initial
        self.total = total
        self.desc = desc or ""
        self.unit = unit
        self.initial = initial
        self.unit_scale = unit_scale
        self.bar_format = bar_format
        self.postfix = None
        self.interval = _INTERVAL
        self.start_t = self.last_t = time.time()
        self.last_n = initial
        self.last_state = None
        self.ema_dn, self.ema_dt = EMA(), EMA()
        self.closed = False

    def __enter__(self) -> "_LogProgress":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def update(self, n: float = 1) -> None:
        self.n += n
        self.refresh()

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs) -> None:
        # Formatted as tqdm.set_postfix does.
        items = {**(ordered_dict or {}), **dict(sorted(kwargs.items()))}
        self.postfix = ", ".join(
            f"{k}={std_tqdm.format_num(v) if isinstance(v, Number) else str(v).strip()}"
            for k, v in items.items()
        )
        if refresh:
            self.refresh()

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        self.postfix = str(s)
        if refresh:
            self.refresh()

    def refresh(self) -> None:
        if time.time() - self.last_t >= self.interval:
            self._log()

    def unpause(self) -> None:
        now = time.time()
        self.start_t += now - self.last_t
        self.last_t, self.last_n = now, self.n

    def write(self, s: str) -> None:
        logger.info("%s", s)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._log(final=True)

    def _log(self, final: bool = False) -> None:
        now = time.time()
        # NumPy scalars (e.g. the TDVP driver's time) as Python numbers.
        n = self.n.item() if hasattr(self.n, "item") else self.n
        total = self.total.item() if hasattr(self.total, "item") else self.total
        state = (n, self.postfix)
        if final and state == self.last_state:
            return
        if n != self.last_n and now > self.last_t:
            self.ema_dn(n - self.last_n)
            self.ema_dt(now - self.last_t)
        line = std_tqdm.format_meter(
            n,
            total,
            now - self.start_t,
            prefix=self.desc,
            unit=self.unit,
            unit_scale=self.unit_scale,
            rate=self.ema_dn() / self.ema_dt() if self.ema_dt.calls else None,
            bar_format=self.bar_format,
            postfix=self.postfix,
            initial=self.initial,
        )
        logger.info("%s", line)
        self.last_t, self.last_n, self.last_state = now, n, state
