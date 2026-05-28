#!/bin/bash
export JAX_PLATFORM_NAME=cpu
python3 -m pytest --cov=netket --cov-append -p no:warnings --color=yes --verbose -n 0 --tb=short "$@" test/sharding/test_sharding.py
