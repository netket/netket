#!/bin/bash
export NETKET_EXPERIMENTAL_SHARDING=1
djaxrun --simple -np 2 python3 -m coverage run -m pytest --clear-cache-every 100 -p no:warnings --color=yes --verbose -n 0 --tb=short $@ test/sharding/test_sharding.py