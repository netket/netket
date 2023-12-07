#!/bin/bash
export NETKET_EXPERIMENTAL_SHARDING=1
export NETKET_EXPERIMENTAL_SHARDING_CPU=2
export JAX_PLATFORM_NAME=cpu
python3 -m pytest -n 0 $@ test
