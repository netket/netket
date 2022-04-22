# Copyright 2022 The NetKet Authors - All rights reserved.
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

import gc
import warnings
from math import ceil, log2
from typing import Optional

from netket.driver import AbstractVariationalDriver
from netket.hilbert import AbstractHilbert


def find_chunk_size(
    driver: AbstractVariationalDriver,
    hilbert: Optional[AbstractHilbert] = None,
    hilbert_size_multiplier: int = 16,
    min_chunk_size: Optional[int] = None,
) -> None:
    """
    Find the maximum chunk size that fits into the memory for a driver.

    Args:
        driver: The driver to be tuned.
        hilbert: The hilbert space used to estimate the initial chunk size.
        hilbert_size_multiplier: A multiplier on the initial chunk size.
        min_chunk_size: The minimum chunk size allowed.

    Note:
        `driver.state.chunk_size` will be modified in-place. Other attributes of
        `driver` may also be modified when `driver._forward_and_backward()` is called.

        This function mainly handles GPU OOM, as CPU OOM in XLA may cause segfault
        and is hard to handle in Python.
    """
    state = driver.state

    # If `state.chunk_size` is already set, we use that as the initial value
    chunk_size = state.chunk_size

    # Otherwise, we initialize `chunk_size` heuristically using
    # `state.n_samples_per_rank` and `hilbert.size`
    if chunk_size is None:
        if hilbert is None:
            if hasattr(driver, "_ham"):
                hilbert = driver._ham.hilbert
            else:
                raise ValueError(
                    "Cannot initialize chunk size because both state.chunk_size "
                    "and hilbert are None"
                )

        chunk_size = state.n_samples_per_rank * hilbert.size * hilbert_size_multiplier

    # Round up to a power of 2
    chunk_size = 2 ** ceil(log2(chunk_size))
    warnings.warn(f"Initialize chunk size to {chunk_size}")
    state.chunk_size = chunk_size

    if min_chunk_size is None:
        min_chunk_size = state.n_samples_per_rank

    while True:
        try:
            # GPU memory is freed when the Python array holding it is collected by GC
            # See https://github.com/google/jax/issues/1222
            gc.collect()

            # Dirty implementation for TDVP
            from netket.experimental.driver import TDVP

            if isinstance(driver, TDVP):
                driver._integrator.step()
            else:
                driver._forward_and_backward()

            break
        except RuntimeError as e:
            if "RESOURCE_EXHAUSTED: Out of memory" not in repr(e):
                raise e

            chunk_size = state.chunk_size // 2
            if chunk_size < min_chunk_size:
                warnings.warn(f"Minimum chunk size {min_chunk_size} reached")
                raise e

            warnings.warn(f"Reduce chunk size to {chunk_size}")
            state.chunk_size = chunk_size
