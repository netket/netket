import re

from typing import Any
from pathlib import Path

import jax
import jax.experimental.multihost_utils

from netket.utils import struct, timing
from netket.utils.optional_deps import import_optional_dependency

from netket._src.callbacks.base import AbstractCallback


class SaveVariationalState(AbstractCallback, mutable=True):
    """
    Callback to save the variational state at fixed intervals. This callback uses the
    `nqxpack <https://github.com/NeuralQXLab/nqxpack>`_ package to save the variational state,
    which allows for portable saving of the variational state.

    If you have problems with the saving/loading of the variational state, open an issue
    over at nqxpack.

    .. warning::
        This callback requires the `nqxpack <https://github.com/NeuralQXLab/nqxpack>`_  package to be installed.
        You can install it with ``uv add nqxpack``. Note that this package is not required by default, and is
        not installed with the main NetKet package.

    .. warning::
        A limitation of nqxpack is that it can only save variational states with models defined inside of a package
        that can be imported. This callback is not compatible with models defined within a script or a notebook,
        as those cannot be imported by nqxpack.

    To load a saved state you can use the ``nqxpack.load`` function, which will return a NetKet variational state object.

    Example usage:
        >>> import netket as nk
        >>> import nqxpack  # doctest: +SKIP
        >>> ...  # doctest: +SKIP
        >>> driver.run(  # doctest: +SKIP
        ...     n_iter=50,
        ...     out="test",
        ...     callback=nk.callbacks.SaveVariationalStateCallback(path="optimization", interval=10),
        ... )
        >>> nqxpack.load("optimization/state_00010.nk")  # doctest: +SKIP

    """

    _path: Path = struct.field(pytree_node=False, serialize=False)
    _interval: int = struct.field(pytree_node=False, serialize=False)
    _file_name_root: str = struct.field(
        pytree_node=False, serialize=False, default="state"
    )
    _max_to_keep: int | None = struct.field(
        pytree_node=False, serialize=False, default=None
    )

    def __init__(
        self,
        path: str | Path,
        interval: float | int,
        *,
        file_name_root: str = "state",
        max_to_keep: int | None = None,
    ):
        r"""
        Constructs the callback to save the variational state at fixed intervals.

        The variational state is saved every ``interval`` iterations in the directory specified by ``path``.
        The file name of the saved state will be of the form ``{file_name_root}_{step:05d}.nk``, where ``step``
        is the iteration number at which the state was saved.

        Args:
            path (str | Path): The path where to save the variational state.
            interval (float | int, optional): The interval at which to save the variational state
            file_name_root (str, optional): The root of the file name to save the variational state. Defaults to "state".
            max_to_keep (int | None, optional): If set, only the most recent ``max_to_keep`` saved states are
                kept on disk; older ones are deleted after each save. Defaults to None (keep all).
        """
        if not isinstance(path, Path):
            path = Path(path)
        self._path = path
        self._interval = interval
        self._file_name_root = file_name_root
        self._max_to_keep = max_to_keep

        if jax.process_index() == 0:
            self._path.mkdir(parents=True, exist_ok=True)

        import_optional_dependency(
            "nqxpack",
            descr="to save the variational state with SaveVariationalStateCallback.",
        )

    def _save(self, step, driver):
        path = self._path / f"{self._file_name_root}_{step:05d}.nk"
        _nqxpack_save(driver.state, path)

        if self._max_to_keep is not None:
            if jax.process_index() == 0:
                pattern = re.compile(rf"^{re.escape(self._file_name_root)}_(\d+)\.nk$")
                saved = sorted(
                    (
                        p
                        for p in self._path.glob(f"{self._file_name_root}_*.nk")
                        if pattern.match(p.name)
                    ),
                    key=lambda p: int(pattern.match(p.name).group(1)),
                )
                for old in saved[: -self._max_to_keep]:
                    old.unlink()
            # Barrier so that other processes do not race ahead (e.g. saving the
            # next checkpoint) while rank 0 is still pruning the directory.
            jax.experimental.multihost_utils.sync_global_devices(
                "SaveVariationalState._save"
            )

    def before_parameter_update(self, step, log_data, driver):
        if step % self._interval == 0:
            self._save(step, driver)

    def on_run_end(self, step, driver):
        self._save(step, driver)


@timing.timed
def _nqxpack_save(state: Any, path: Path):
    nqxpack = import_optional_dependency(
        "nqxpack",
        descr="to save the variational state with SaveVariationalStateCallback.",
    )
    if jax.process_index() == 0 and path.exists():
        print(f"Warning: file {path} already exists and will be overwritten.")
    nqxpack.save(state, path)
