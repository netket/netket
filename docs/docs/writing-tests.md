# Writing Tests

NetKet's tests are written using the [PyTest](https://docs.pytest.org/en/stable/contents.html)
framework.
In particular, we make extensive use of [`pytest.parametrize`](https://docs.pytest.org/en/stable/parametrize.html#pytest-mark-parametrize-parametrizing-test-functions) in order to run the same
test functions on different input types. 
We also use often [`pytest.fixture`](https://docs.pytest.org/en/stable/fixture.html#what-fixtures-are) 
to initialize expensive objects only once among different tests.

## Test structure and common files.

NetKet's `Test` folder is a python module because every folder (aka, submodule) has an
`__init__.py` file.
Tests are grouped in submodules according to the NetKet's submodule they tests.
If you add a new submodule, you should add a new submodule to the tests too.

If you add a new file to netket, if might be a good idea to split it's tests into a new file in
the relevant submodule, too.

Common functions and methods used throughout our testing infrastructure are defined in the file
`test/common.py` and every test is expected to use them if necessary. 
Some common fixtures are also defined inside `test/conftest.py` and are available to all tests.
Those do not need to be imported explicitly, as pytest will take care of it.

## Test Markers

NetKet uses pytest markers to categorize tests. Currently, the following markers are defined:

- `@pytest.mark.slow`: Marks tests that take a long time to run (typically >15 seconds). These tests are excluded from CI to keep build times reasonable, but are run locally during development.

### Running Tests

To run all tests including slow ones:
```bash
pytest test/
```

To run tests excluding slow ones (same as CI):
```bash
pytest -m "not slow" test/
```

To run only slow tests:
```bash
pytest -m "slow" test/
```
