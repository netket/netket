### NetKet Sharding Tests

This folder contains tests for running NetKet on multiple jax devices.

They are kept separate from the normal tests as they need to be run with different flags (some of those set in pyproject.toml are incompatible and I haven't found a reliable way to override them)

In particular it contains scripts to run:

- run the normal tests on 2 cpu devices (run_standard_tests_with_sharding.sh)
- run specific tests for sharding on 2 cpu devices (run_test_sharding.sh)
- run specific tests for sharding on 2 separate processes (run_test_sharding_distributed.sh)

They need to be run from the root folder of the netket repository, e.g. with `./test_sharding/run_test_sharding.sh`
