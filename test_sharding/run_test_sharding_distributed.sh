#!/bin/bash
djaxrun --simple -np 2 python3 -m coverage run -m pytest --clear-cache-every 100 -p no:warnings --color=yes --verbose -n 0 --tb=short "$@" test/sharding/test_sharding.py test/logging/test_hdf5_log.py
