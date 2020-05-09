# Contributing guidelines

Thank you for getting involved in NetKet. It is only thanks to the involvement of
a large community of developers that open-source software like NetKet can thrive.
Please take the time to read the following guidelines, to ensure that your contributions
are in line with the general goals of this project and its requirements.  

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read [contributing guidelines](CONTRIBUTING.md).
- Read [Code of Conduct](CODE_OF_CONDUCT.md).
- Check if your changes are consistent with the [guidelines](CONTRIBUTING.md#general-guidelines-and-philosophy-for-contribution).
- Check if your changes are consistent with the [Python](CONTRIBUTING.md#python-coding-style) code style and run the Black code formatter as needed.
- Run [Unit Tests](CONTRIBUTING.md#running-unit-tests).

## How to become a contributor and submit your own code

### Contributing code

If you have improvements to NetKet, or want to get involved in one of our [Open Issues](https://github.com/netket/netket/issues) send us your pull requests! For those just getting started, Github has a [howto](https://help.github.com/articles/using-pull-requests/).

NetKet team members will be assigned to review your pull requests. Once the pull requests are approved and pass continuous integration checks, we will merge the pull requests.

If you want to contribute but you're not sure where to start, take a look at the
[issues with the "help wanted" label](https://github.com/netket/netket/labels/help%20wanted).
These are issues that we believe are particularly well suited for outside
contributions. If you decide to start on an issue, leave a comment so that other people know that
you're working on it. If you want to help out, but not alone, use the issue comment thread to coordinate.

### Contribution guidelines and standards

Before sending your pull request for
[review](https://github.com/netket/netket/pulls),
make sure your changes are consistent with the guidelines and follow the
NetKet coding style.

#### General guidelines and philosophy for contribution

* Include unit tests when you contribute new features, as they help to
  a) prove that your code works correctly, and b) guard against future breaking
  changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs
  usually indicates insufficient test coverage.
* When you contribute a new feature to NetKet, the maintenance burden is (by
  default) transferred to the NetKet team. This means that benefit of the
  contribution must be compared against the cost of maintaining the feature.

#### License

Include a license at the top of new files.

#### Python coding style

Python code should follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
For consistency, we use the [Black](https://github.com/python/black) code formatter, which you can install in your Python environment using `pip install black`.
If you edit Python code, you should run Black on the affected files.
On the command line, this can be done via
```bash
# to reformat a specific file
black Path/To/The/file.py
# to reformat all files below the specified directories
black Test/ Examples/
```
before creating a pull request.
There are other options of running Black: For example, an [Atom package](https://atom.io/packages/python-black) is available and Black is integrated into the [VS Code Python extension](https://code.visualstudio.com/docs/python/editing#_formatting).


#### Including new unit tests

Contributions implementing new features **must** include associated unit tests.
Unit tests in NetKet are based on [pytest](https://docs.pytest.org/en/latest/) and are located in the directory `Test`.

#### Running unit tests
Unit tests can be run locally doing

```bash
pytest Test --verbose
```

this will highlight cases where tests are failing. These must be addressed before any pull request can be merged on stable branches.
 
#### Test MPI code

For code that relies on MPI operations, there are units tests in the `Test_MPI` subfolder.
If you add MPI-related code, please also add corresponding unit tests to that directory.

The MPI tests can be run with
```bash
mpirun -np 2 pytest Test_MPI --verbose
```
Note that this will print the test output twice, as `pytest` itself is not aware of being run with MPI.
Running these tests also provides a quick way to check if your MPI setup is working.
If `test_is_running_with_multiple_procs` or `test_mpi_setup` is failing, you should check your MPI configuration,
for example whether `mpi4py` is compiled against the correct MPI libraries for your machine.
