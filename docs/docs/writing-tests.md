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

## Tests and MPI

Tests are expected to run with or without mpi4py installed, under MPI and not under MPI.
Therefore you should never import `mpi4py` or `mpi4jax` in the global test module, but only
inside individual tests.

Tests not testing MPI-related fucntionality should be skipped when executed under MPI.
To mark a whole module to be skipped under mpi, you can define the following variable

```python
# test/hilbert/test_tensor.py

from .. import common

pytestmark = common.skipif_mpi


def test_tensors():
	...
```

Alternatively, you can skip individual testts by decorating them with `common.skipif_mpi`.

```python

@common.skipif_mpi
def my_serial_test():
	...
```

To execute a test only when run with MPI, you can use the decorator `common.onlyif_mpi` in the 
same way as shown in the two examples above.

If, inside your tests, you need to run some NetKet functions with MPI disabled, for example to 
check that the MPI code gives the same result as the non-MPI code, you can use the 
`netket_disable_mpi` object as follows:

```python
from .. import common

@onlyif_mpi
def test_matmul():
	...
	x_mpi = A@v

	with common.netket_disable_mpi():
		x_serial = A@v

		np.testing.assertr_allclose(x_mpi, x_serial)
```

For simplicity, you can also use the fixtures `_mpi_size`, `_mpi_rank` and `_mpi_comm` as inputs to your test 
functions to get easily those informations. See the example below:

```python

# run this with mpi and not
def test_mpi_things(_mpi_rank, _mpi_size):
	if _mpi_size == 1:
		# mpi disabled
	else:
		# mpi enabled

```