#!/bin/bash
mpirun -host localhost:2 -np 2 -x NETKET_EXPERIMENTAL_SHARDING=1 -x NETKET_MPI_WARNING=0 python3 -m pytest -p no:warnings --color=yes --verbose -n 0 --tb=short $@ test_sharding/test_sharding_distributed.py
