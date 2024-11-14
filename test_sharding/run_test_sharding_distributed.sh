#!/bin/bash
export NETKET_EXPERIMENTAL_SHARDING=1
export NETKET_MPI_WARNING=0 
mpirun -np 2 python3 -m coverage run -m pytest --jax-distributed-gloo --clear-cache-every 100 -p no:warnings --color=yes --verbose -n 0 --tb=short $@ test/sharding/test_sharding.py