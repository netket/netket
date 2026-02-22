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
       class VMC(AbstractVariationalDriver):
            ...

        @get_forward_operator.dispatch
        def _get_forward_operator_vmc(driver: VMC):
            return driver.hamiltonian
    """
    raise NotImplementedError(
        f"get_forward_operator not implemented for {type(driver)}"
    )


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
    values and store them in the callback object. If you reuse the same callback object
    in a subsequent run, it will reuse the same chunk sizes.

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

        class MyDriver(AbstractVariationalDriver):
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

    def on_run_start(self, step, driver):
        if jax.process_count() == 0:
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
                self.sampler_chunk_size = sampler_chunk_size
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
            self.chunk_size = vstate_chunk_size
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
            self.chunk_size_bwd = chunk_size_bwd

        print("Chunk size tuning complete:")
        if set_sampler_chunk_size:
            print(f"  {sampler_chunk_size = }")
        if set_vstate_chunk_size:
            print(f"  {vstate_chunk_size = }")
        if set_bwd_chunk_size:
            print(f"  {chunk_size_bwd = }")
