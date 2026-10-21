import copy
import zlib

import jax
from jax.errors import JaxRuntimeError

from netket.utils import struct, timing, dispatch

from netket._src.callbacks.base import AbstractCallback


@dispatch.dispatch
def get_forward_operator(driver):
    """
    Returns an operator whose expectation value computes the forward
    pass for the driver.

    This is a dispatch function: register new overloads with
    ``@get_forward_operator.dispatch`` for custom driver types.

    Example:
       class VMC(AbstractOptimizationDriver):
            ...

        @get_forward_operator.dispatch
        def _get_forward_operator_vmc(driver: VMC):
            return driver.hamiltonian
    """
    raise NotImplementedError(
        f"get_forward_operator not implemented for {type(driver)}"
    )


def compute_memory_config_hash(driver, hints) -> int:
    """A hash of what the chunk sizes depend on: the devices, the number of samples
    and chains, the structure, shapes and dtypes of the parameters, and ``hints``,
    the chunk sizes tuning starts from."""
    state = driver.state
    memory_config = (
        jax.device_count(),
        jax.devices()[0].device_kind,
        getattr(state, "n_samples", None),
        getattr(getattr(state, "sampler", None), "n_chains", None),
        jax.tree.structure(state.parameters),
        [(x.shape, x.dtype) for x in jax.tree.leaves(state.parameters)],
        hints,
    )
    return zlib.crc32(repr(memory_config).encode())


class AutoChunkSize(AbstractCallback):
    """
    Automatically tunes the chunk size for the (i) sampler, (ii) forward pass and
    (iii) backward pass of a driver and a variational state in order to avoid OOM errors.

    This callback is useful when the optimal chunk size is not known in advance,
    and the user wants to avoid manually tuning it.

    The callback will try to run the sampler, forward pass and backward pass with
    decreasing chunk sizes until it finds one that fits in memory. The chunk size
    is halved on each failure, down to ``minimum_chunk_size``.

    After the largest chunk size that works is found, the callback will print those
    values and store them in the callback object, leaving the initial chunk sizes it
    was given unchanged. A later run with the same callback object reuses them without
    tuning again, unless the devices, the number of samples or chains, the shapes and
    dtypes of the parameters, or the initial chunk sizes changed: then it tunes again,
    starting from the initial chunk sizes. Passed as a persistent callback
    (``callbacks=`` of the driver), the chunk sizes are saved with a checkpoint, so a
    resumed run does not tune again either.

    .. rubric:: Supporting custom drivers

    To tune the forward pass, the callback needs to know which operator to use to
    estimate the loss. This is determined by the :func:`get_forward_operator`
    dispatch function, which is already registered for the built-in drivers
    (:class:`~netket.driver.VMC`, :class:`~netket.driver.VMC_SR`,
    :class:`~netket.driver.Infidelity_SR`).

    If you implement a custom driver, you must register a new overload so that
    ``AutoChunkSize`` knows how to probe the forward pass:

    .. code-block:: python

        from netket._src.callbacks.auto_chunk_size import get_forward_operator

        class MyDriver(AbstractOptimizationDriver):
            def __init__(self, hamiltonian, ...):
                self._ham = hamiltonian
                ...

        @get_forward_operator.dispatch
        def _(driver: MyDriver):
            return driver._ham

    The overload should return an operator whose expectation value exercises the
    same forward-pass code path as the actual optimisation step. The registration
    is best placed at the bottom of the file that defines the driver class, so
    it takes effect as soon as the driver is imported.
    """

    minimum_chunk_size: int = struct.field(pytree_node=False, serialize=False)
    sampler_chunk_size: int | None = struct.field(pytree_node=False, serialize=False)
    chunk_size: int | None = struct.field(pytree_node=False, serialize=False)
    chunk_size_bwd: int | None = struct.field(pytree_node=False, serialize=False)
    #: The chunk sizes found by tuning (-1: keep the driver's own), and the memory
    #: configuration hash they were found for (-1: not tuned); saved with a checkpoint.
    _inferred_sampler_chunk_size: int = struct.field(pytree_node=False, serialize=True)
    _inferred_chunk_size: int = struct.field(pytree_node=False, serialize=True)
    _inferred_chunk_size_bwd: int = struct.field(pytree_node=False, serialize=True)
    _memory_config_hash: int = struct.field(pytree_node=False, serialize=True)

    def __init__(
        self,
        sampler_chunk_size: int | None = None,
        chunk_size: int | None = None,
        chunk_size_bwd: int | None = None,
        minimum_chunk_size: int = 1,
    ):
        """
        Initialize the callback.

        Args:
            sampler_chunk_size: The initial chunk size to use for the sampler (default: None).
                If None, the callback will try to find a chunk size that works starting from
                the number of chains per rank.
            chunk_size: The initial chunk size to use for the forward pass (default: None).
                If None, the callback will try to find a chunk size that works starting from
                the number of samples.
            chunk_size_bwd: The initial chunk size to use for the backward pass (default: None).
                If None, the callback will try to find a chunk size that works starting from
                the number of samples.
            minimum_chunk_size: The minimum chunk size to use (default: 1). If the chunk size
                is decreased to this value and it still fails, the callback will raise an error.
        """
        self.sampler_chunk_size = sampler_chunk_size
        self.chunk_size = chunk_size
        self.chunk_size_bwd = chunk_size_bwd

        self.minimum_chunk_size = minimum_chunk_size
        self._inferred_sampler_chunk_size = -1
        self._inferred_chunk_size = -1
        self._inferred_chunk_size_bwd = -1
        self._memory_config_hash = -1

    def __process_deserialization_state__(self, state):
        # TODO: backward compatibility, for checkpoints saved before October 2026
        # (without these fields, or with them packed in `_tuned`): tune again.
        not_tuned = {
            "_inferred_sampler_chunk_size": -1,
            "_inferred_chunk_size": -1,
            "_inferred_chunk_size_bwd": -1,
            "_memory_config_hash": -1,
        }
        state = {name: value for name, value in state.items() if name != "_tuned"}
        return {**not_tuned, **state}

    def on_run_start(self, step, driver):
        hints = (self.sampler_chunk_size, self.chunk_size, self.chunk_size_bwd)
        memory_config_hash = compute_memory_config_hash(driver, hints)
        if self._memory_config_hash != memory_config_hash:
            # Probe a copy: probing samples and computes an update, which must not
            # change what the run computes.
            probe = driver.replace(_variational_state=copy.copy(driver.state))
            (
                self._inferred_sampler_chunk_size,
                self._inferred_chunk_size,
                self._inferred_chunk_size_bwd,
            ) = (-1 if size is None else size for size in self._tune(probe))
            self._memory_config_hash = memory_config_hash
        state = driver.state
        sampler_chunk_size = self._inferred_sampler_chunk_size
        if sampler_chunk_size != -1 and hasattr(state, "sampler"):
            # Not through the `sampler` setter, which would draw new chains.
            state._sampler = state.sampler.replace(chunk_size=sampler_chunk_size)
        if self._inferred_chunk_size != -1 and hasattr(state, "chunk_size"):
            state.chunk_size = self._inferred_chunk_size
        if self._inferred_chunk_size_bwd != -1 and hasattr(driver, "chunk_size_bwd"):
            driver.chunk_size_bwd = self._inferred_chunk_size_bwd

    def _tune(self, driver) -> tuple[int | None, int | None, int | None]:
        """Tunes ``driver``, starting from the initial chunk sizes, and returns the
        sampler, forward and backward chunk sizes found (None: keep the driver's)."""
        if jax.process_index() == 0:
            print("Automatically tuning chunk size....")

        # Try to check if sampling works
        set_sampler_chunk_size = False
        sampler_chunk_size = None

        if hasattr(driver.state, "sampler"):
            if hasattr(driver.state.sampler, "chunk_size"):
                if self.sampler_chunk_size is not None:
                    sampler_chunk_size = self.sampler_chunk_size
                    driver.state.sampler = driver.state.sampler.replace(
                        chunk_size=self.sampler_chunk_size
                    )
                while True:
                    try:
                        with timing.timed_scope(force=True) as timer:
                            timer.block_until_ready(driver.state.samples)
                    except JaxRuntimeError:
                        set_sampler_chunk_size = True
                        sampler_chunk_size = driver.state.sampler.chunk_size
                        if sampler_chunk_size is None:
                            sampler_chunk_size = driver.state.sampler.n_chains_per_rank

                        print(
                            f"- Sampling failed with {sampler_chunk_size = }, trying with {sampler_chunk_size // 2 }"
                        )
                        sampler_chunk_size = sampler_chunk_size // 2
                        if sampler_chunk_size < self.minimum_chunk_size:
                            raise ValueError("Sampling failed with minimum chunk size")
                        driver.state.sampler = driver.state.sampler.replace(
                            chunk_size=sampler_chunk_size
                        )
                    else:
                        print(f"- Sampling succeeded with {sampler_chunk_size = }")
                        break
        # Try to check if forward pass works
        set_vstate_chunk_size = False
        vstate_chunk_size = None
        if hasattr(driver.state, "chunk_size"):
            operator = get_forward_operator(driver)
            if self.chunk_size is not None:
                vstate_chunk_size = self.chunk_size
                driver.state.chunk_size = self.chunk_size
            while True:
                try:
                    with timing.timed_scope(force=True) as timer:
                        timer.block_until_ready(driver.estimate(operator))

                except JaxRuntimeError:
                    set_vstate_chunk_size = True
                    vstate_chunk_size = driver.state.chunk_size
                    if vstate_chunk_size is None:
                        vstate_chunk_size = driver.state.n_samples

                    print(
                        f"- Forward pass failed with {vstate_chunk_size = }, trying with {vstate_chunk_size // 2 }"
                    )
                    vstate_chunk_size = vstate_chunk_size // 2
                    if vstate_chunk_size < self.minimum_chunk_size:
                        raise ValueError("Forward pass failed with minimum chunk size")

                    driver.state.chunk_size = vstate_chunk_size
                else:
                    print(f"- Forward pass succeeded with {vstate_chunk_size = }")
                    break
        # backward
        set_bwd_chunk_size = False
        chunk_size_bwd = None
        if hasattr(driver, "chunk_size_bwd"):
            driver.reset_step()
            if self.chunk_size_bwd is not None:
                chunk_size_bwd = self.chunk_size_bwd
                driver.chunk_size_bwd = self.chunk_size_bwd

            while True:
                try:
                    with timing.timed_scope(force=True) as timer:
                        timer.block_until_ready(driver.compute_loss_and_update())

                except JaxRuntimeError:
                    set_bwd_chunk_size = True
                    chunk_size_bwd = driver.chunk_size_bwd
                    if chunk_size_bwd is None:
                        chunk_size_bwd = driver.state.n_samples

                    print(
                        f"- Optimisation step failed with {chunk_size_bwd = }, trying with {chunk_size_bwd // 2 }"
                    )
                    chunk_size_bwd = chunk_size_bwd // 2
                    if chunk_size_bwd < self.minimum_chunk_size:
                        raise ValueError(
                            "Optimisation pass failed with minimum chunk size"
                        )
                    driver.chunk_size_bwd = chunk_size_bwd
                else:
                    print(f"- Optimisation step succeeded with {chunk_size_bwd = }")
                    break

        print("Chunk size tuning complete:")
        if set_sampler_chunk_size:
            print(f"  {sampler_chunk_size = }")
        if set_vstate_chunk_size:
            print(f"  {vstate_chunk_size = }")
        if set_bwd_chunk_size:
            print(f"  {chunk_size_bwd = }")
        return sampler_chunk_size, vstate_chunk_size, chunk_size_bwd
