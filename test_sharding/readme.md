### NetKet multi-process tests

The test suite runs on 2 CPU devices by default (see `test/conftest.py`), so the
multi-device code paths are exercised by a plain `pytest` run. Use
`pytest --devices N` (or `NETKET_TEST_DEVICES=N`) to pick another device count,
e.g. `--devices 1` for the single-device code paths.

This folder only contains the script that runs tests across 2 separate processes
with `djaxrun` (`run_test_sharding_distributed.sh`). It must be run from the root
folder of the netket repository, e.g. with `./test_sharding/run_test_sharding_distributed.sh`.
