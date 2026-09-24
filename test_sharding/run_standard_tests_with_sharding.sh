#!/bin/bash
export NETKET_EXPERIMENTAL_SHARDING_CPU=2
export JAX_PLATFORM_NAME=cpu
# Each xdist worker is its own process and gets its own pair of CPU devices, so
# this parallelises safely. 2 workers rather than `auto`/`logical` because every
# worker already spreads itself over NETKET_EXPERIMENTAL_SHARDING_CPU devices.
# Override by passing another -n after the script name.
python3 -m pytest --cov=netket --cov-append -n 2 "$@" test
