# Contributing to NetKet

Everyone can contribute to NetKet: it is only thanks to the involvement of a large community of developers that open-source 
software like NetKet can thrive.
There are several ways to contribute, including:

- Answering questions on NetKet's [discussions page](https://github.com/netket/netket/discussions)
- Improving or expanding NetKet's [documentation](https://www.netket.org/docs/getting_started.html)
- Improving or expanding NetKet's [tutorial notebooks](https://www.netket.org/tutorials.html)
- Contributing to NetKet's [code-base](https://github.com/netket/netket/)

## Ways to contribute

We welcome pull requests, in particular for those issues marked with
[contributions welcome](https://github.com/netket/netket/labels/contributor%20welcome) or
[good first issue](https://github.com/netket/netket/labels/good%20first%20issue). 
You can also have a look at our [RoadMap](https://github.com/netket/netket/issues/559) for ideas
on things that we would like to implement but haven't done so yet.
It is generally regarded good behaviour to let other know of your plan to contribute a feature.

For other proposals, we ask that you first open a GitHub
[Issue](https://github.com/netket/netket/issues/new/choose) or
[Discussion](https://github.com/netket/netket/discussions)
to seek feedback on your planned contribution.

The NetKet project follows the [Contributor Covenant Code of Conduct](http://contributor-covenant.org/version/1/4/).

## Contributing code using pull requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Fork the NetKet repository by clicking the **Fork** button on the
   [repository page](http://www.github.com/netket/netket). This creates
   a copy of the NetKet repository in your own account.

2. Install Python >=3.9 locally in order to run tests.

3. `pip` installing your fork from source with the development dependencies. 
   This allows you to modify the code and immediately test it out:

   ```bash
   git clone https://github.com/YOUR_USERNAME/netket
   cd netket
   pip install -e '.[dev]'  # Installs NetKet from the current directory in editable mode.
   pre-commit install # Install the pre-commit hook that checks your commit for good formatting
   ```

4. Add the NetKet repo as an upstream remote, so you can use it to sync your
   changes.

   ```bash
   git remote add upstream http://www.github.com/netket/netket
   ```

5. Create a branch where you will develop from:

   ```bash
   git checkout -b name-of-change
   ```

   And implement your changes.

6. You can make sure your code passes some 'preliminary' tests for very simple errors or code style by running (the flag `--fix` will automatically fix most errors).

   ```bash
   ruff check --fix
   ```

7. Make sure the tests pass by running the following command from the top of
   the repository:

   ```bash
   pytest -n auto test/
   ```

   If your code contribution touches parts of NetKet that are expected to work under MPI, such
   the computation of expectation values, initialization of parameters or lazy matrix-vector products
   you should also check that tests pass under MPI by running pytest under MPI. When doing so, you
   should also disable pytest parallelization.

   ```bash
   mpirun -np 2 pytest -n0 test/
   ```

   NetKet's test suite is quite large, so if you know the specific test file or folder that covers your changes, you can limit the tests to that, as shown in the example below:

   ```bash
   pytest -n auto test/hilbert
   ```

   You can narrow the tests further by using the `pytest -k` flag to match particular test names, as shown in this example here:

   ```bash
   pytest -n auto test/hilbert -k 'Spin'
   ```

8. To make sure that your code is properly formatted and passes some 'preliminary checks' you should also run black (and ruff) by running
   ```bash
   pre-commit run --all-files 
   ```

9. Once you are satisfied with your change, create a commit as follows (
   [how to write a commit message](https://chris.beams.io/posts/git-commit/)):

   ```bash
   git add file1.py file2.py ...
   git commit -m "Your commit message"
   ```
   You can also use a graphical git client that simplifies a lot using Git.
   We suggest [GitKraken](https://www.gitkraken.com/) or [SourceTree](https://www.sourcetreeapp.com/).

   Finally, push your commit on your development branch and create a remote 
   branch in your fork that you can use to create a pull request from:

   ```bash
   git push --set-upstream origin name-of-change
   ```

10. Create a pull request from the NetKet repository and send it for review.
   Check the {ref}`pr-checklist` for considerations when preparing your PR, and
   consult [GitHub Help](https://help.github.com/articles/about-pull-requests/)
   if you need more information on using pull requests.


(pr-checklist)=

## NetKet pull request checklist

As you prepare a NetKet pull request, here are a few things to keep in mind:

### Single-change commits and pull requests

It is considered good practice that every git commit sent for review is a self
-contained, single change with a descriptive message. 
This helps with review and with identifying or reverting changes if
issues are uncovered later on.

**Pull requests ideally comprise a single git commit.** (In some cases, for
instance for large refactors or internal rewrites, they may contain several.)
In preparing a pull request for review, you may want to squash together
multiple commits. 

We don't really enforce or require those rules, but they are welcome additions.

### Python coding style

Code should follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
For consistency, we use the [Black](https://github.com/python/black) code formatter.
The precise version we use to check formatting will be installed when you install NetKet's development
dependencies as outlined above.

If you edit Python code, you should run Black on the affected files.
On the command line, this can be done via
```bash
# to reformat a specific file
black Path/To/The/file.py
# to reformat all files below the specified directories
black test/ Examples/
```
before creating a pull request.
There are other options of running Black: For example, an [Atom package](https://atom.io/packages/python-black) is available and Black is integrated into the [VS Code Python extension](https://code.visualstudio.com/docs/python/editing#_formatting).

If you have installed [pre-commit](https://pre-commit.com/), black will be run automatically before you commit or
if you run the command below.

```bash
pip install pre-commit
pre-commit run --all
```

### Docstrings and Type Hints

If you have written new methods, we expect those to be documented with Type Hints for
their inputs and output using standard Python types from the `typing` standard library
and NetKet's custom types from {code}`netket.utils.types` for more advanced objects such
as PyTrees, Arrays and data types.

Your new methods and functions should always be documented with a docstring in the 
[Napoleon Google's format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

### Visibility of public API

If you have added an object that is supposed to be used by end-users and will be part 
of the public API, we expect you to add this object to the documentation by adding
the relevant classes/functions to the auto-generated API in `docs/api.rst` and eventually
by writing a new documentation page.

### Full GitHub test suite

Your PR will automatically be run through a full test suite on GitHub CI, which
covers a range of Python versions, dependency versions, and configuration options.
It's normal for these tests to turn up failures that you didn't catch locally; to
fix the issues you can push new commits to your branch.
