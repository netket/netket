#!/bin/bash
# to run on our self-hosted runner with 2 gpu vm's
mpirun --hostfile /etc/hostfile -x NETKET_EXPERIMENTAL_SHARDING=1 -x NETKET_MPI_WARNING=0 python3 -m pytest -p no:warnings --color=yes --verbose -n 0 --tb=short $@ test_sharding/test_sharding_distributed.py
