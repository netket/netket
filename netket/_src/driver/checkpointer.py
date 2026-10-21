import abc
from typing import NamedTuple

import jax
import numpy as np
from flax import serialization

from netket._src.callbacks.base import StopRun
from netket.utils.struct import Pytree
from netket.vqs import VariationalState


class Checkpoint(NamedTuple):
    """A checkpoint read back by :meth:`AbstractCheckpointer.read` (experimental)."""

    step: int
    #: The saved state, as :func:`checkpoint_state` arranges it.
    state: dict
    #: What the checkpointer recorded with it (e.g. the run segment, versions).
    metadata: dict


class AbstractCheckpointer(abc.ABC):
    """
    Saves a driver run, and restores it when the run is started again (e.g.
    after a crash or a preemption). Pass one to
    :meth:`~netket.driver.AbstractDriver.run` as ``checkpointer=``.

    On every process, a run calls:

    - :meth:`restore`, after the callbacks' ``on_run_prep_allocate`` and before
      any other hook;
    - after each step, :meth:`should_stop`, then :meth:`should_save`; if that
      returns ``True``, every callback's ``on_checkpoint_save``, then :meth:`save`;
    - :meth:`wait` before ``on_run_end``, and :meth:`close` however the run ends.

    The checkpoint of step ``s`` holds the run after ``s`` complete steps; none
    is saved before the first step. :meth:`latest_step` and :meth:`read` work
    without a driver, e.g. to seed a new run (see
    :func:`netket.checkpoint.load_checkpoint_state`).

    Experimental: this interface may change without a deprecation period.
    """

    @abc.abstractmethod
    def restore(self, driver, *, start_step: int, target_step: int) -> dict | None:
        """
        The state a run from ``start_step`` to ``target_step`` resumes from,
        arranged as :func:`checkpoint_state` of ``driver``, or ``None`` to start
        fresh. Raises if the checkpoint does not belong to this run. ``driver``
        is only to be read.
        """

    @abc.abstractmethod
    def latest_step(self) -> int | None:
        """The step of the latest saved checkpoint, or ``None`` if there is none."""

    @abc.abstractmethod
    def read(
        self, step: int | None = None, *, target: dict | None = None
    ) -> Checkpoint | None:
        """
        Reads the checkpoint of ``step`` (default: the latest), with no driver;
        ``None`` if there is none.

        Without ``target``, the arrays come back as stored. With ``target``, a
        :func:`checkpoint_state` of a live driver, they come back with its
        structure, dtypes and sharding.
        """

    @abc.abstractmethod
    def should_save(self, step: int, *, last: bool) -> bool:
        """
        Whether to save after step ``step``; ``last`` if the run stops or ends
        there. It must return the same on every process.
        """

    @abc.abstractmethod
    def save(self, step: int, state: dict) -> None:
        """
        Saves ``state``, the :func:`checkpoint_state` of the driver, as the
        checkpoint of ``step``. It may write in the background, but must not
        depend on ``state`` once it returns: the run goes on.
        """

    def should_stop(self, step: int) -> StopRun | None:
        """
        A :class:`~netket.callbacks.StopRun` to stop the run after step ``step``
        (e.g. on a preemption notice), or ``None``. Asked on every process; one
        process's stop stops them all. Default: ``None``.
        """

    def wait(self) -> None:
        """Waits until every save is written; raises the error of one that failed."""

    def close(self, cause: BaseException | None = None) -> None:
        """
        Releases what the run used; called once however the run ends, also after
        a failed :meth:`restore`. ``cause`` is what ended the run: a
        :class:`~netket.callbacks.StopRun`, a ``KeyboardInterrupt`` or the error;
        ``None`` if it completed. Default: no-op.
        """


def checkpoint_state(driver) -> dict:
    """
    The state of ``driver`` a checkpointer saves: its
    ``flax.serialization.to_state_dict``, arranged as ``{"driver": its own fields,
    "variational_state": ..., "callbacks": [one per persistent callback]}``.
    """
    return group_state(serialization.to_state_dict(driver))


def group_state(state_dict: dict) -> dict:
    """Arranges a driver's ``to_state_dict`` as :func:`checkpoint_state` does."""
    driver = dict(state_dict)
    variational_state = driver.pop("state")
    callbacks = driver.pop("_callbacks")
    return {
        "driver": driver,
        "variational_state": variational_state,
        "callbacks": [callbacks[key] for key in sorted(callbacks, key=int)],
    }


def flatten_state(state: dict) -> dict:
    """The driver's ``to_state_dict`` from a :func:`checkpoint_state`."""
    return {
        **state["driver"],
        "state": state["variational_state"],
        "_callbacks": {str(i): s for i, s in enumerate(state["callbacks"])},
    }


def load_checkpoint_state(driver, state: dict, *, only=None):
    """
    Loads ``state``, a :func:`checkpoint_state` (e.g. ``checkpointer.read().state``),
    into ``driver`` in place.

    With ``only``, loads just those parts of it, given as paths such as
    ``"variational_state"`` or ``"variational_state/variables"``, and keeps the
    driver's own for the rest. ``only=("variational_state/variables",)`` seeds a
    new run with the parameters of an old one: its step, optimizer, callbacks,
    sampler and batch size stay those of the new driver.
    """
    if only is not None:
        own = checkpoint_state(driver)
        for path in only:
            *parents, key = path.split("/")
            source, destination = state, own
            for parent in parents:
                source, destination = source[parent], destination[parent]
            destination[key] = _align_keys(source[key], destination[key])
        state = own
    load_state_inplace(driver, flatten_state(state))


def _align_keys(state, like):
    """
    ``state`` with the key types of ``like``: a store keyed by strings gives back
    ``"0"`` for the integer key ``0`` of an nnx layer list.
    """
    if not isinstance(state, dict) or not isinstance(like, dict):
        return state
    keys = {str(key): key for key in like}
    aligned = {}
    for key, value in state.items():
        key = keys.get(str(key), key)
        aligned[key] = _align_keys(value, like.get(key))
    return aligned


def load_state_inplace(target, state):
    """
    Loads ``state``, from ``flax.serialization.to_state_dict(target)``, into
    ``target`` in place.

    :func:`flax.serialization.from_state_dict` returns a new object, but a
    restored driver must stay the same object, and so must the variational
    state and the callbacks it holds: the caller and the callback list of the
    run refer to them. ``state`` is deserialized whole before ``target``
    changes, so a state that does not fit raises with ``target`` untouched.
    """
    _write_back(target, serialization.from_state_dict(target, state))


def _write_back(target, restored):
    for name, new in vars(restored).items():
        current = getattr(target, name, None)
        if current is not None and current is not new and _is_held(current, new):
            _write_back(current, new)
        elif isinstance(current, tuple | list) and _holds(current, new):
            # The persistent callbacks: the callback list of a run holds them.
            for current_item, new_item in zip(current, new, strict=True):
                if current_item is not new_item and _is_held(current_item, new_item):
                    _write_back(current_item, new_item)
        else:
            setattr(target, name, new)


def _is_held(current, new) -> bool:
    """
    Whether ``current`` is to be updated in place rather than replaced: a mutable
    :class:`Pytree` (the driver, a callback) or a variational state. Anything else
    (arrays, immutable pytrees such as a sampler state) is replaced.
    """
    if type(current) is not type(new):
        return False
    if isinstance(current, jax.Array | np.ndarray | np.generic):
        return False
    if not getattr(current, "__dict__", None):
        return False
    if isinstance(current, Pytree):
        return type(current)._pytree__class_is_mutable
    return isinstance(current, VariationalState)


def _holds(current, new) -> bool:
    return (
        isinstance(new, tuple | list)
        and len(current) == len(new)
        and any(_is_held(c, n) for c, n in zip(current, new, strict=True))
    )
