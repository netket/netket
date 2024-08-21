#!/bin/bash
export NETKET_EXPERIMENTAL_SHARDING=1
export NETKET_MPI_WARNING=0 
mpirun -np 2 python3 -m pytest --jax-distributed-gloo --cov=netket --cov-append -p no:warnings --color=yes --verbose -n 0 --tb=short $@ test/sharding/test_sharding.py