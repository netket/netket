# Contributing guidelines

Thank you for getting involved in NetKet. It is only thanks to the involvement of
a large community of developers that open-source software like NetKet can thrive.
Please take the time to read the following guidelines, to ensure that your contributions
are in line with the general goals of this project and its requirements.  

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read [contributing guidelines](CONTRIBUTING.md).
- Read [Code of Conduct](CODE_OF_CONDUCT.md).
- Check if my changes are consistent with the [guidelines](CONTRIBUTING.md#general-guidelines-and-philosophy-for-contribution).
- Changes are consistent with the [Coding Style](CONTRIBUTING.md#c-coding-style).
- Run [Unit Tests](CONTRIBUTING.md#running-unit-tests).

## How to become a contributor and submit your own code

### Contributing code

If you have improvements to NetKet, or want to get involved in one of our [Challenges](https://www.netket.org/challenges/home/) send us your pull requests! For those
just getting started, Github has a [howto](https://help.github.com/articles/using-pull-requests/).

NetKet team members will be assigned to review your pull requests. Once the pull requests are approved and pass continuous integration checks, we will merge the pull requests.

If you want to contribute but you're not sure where to start, take a look at the
[issues with the "help wanted" label](https://github.com/netket/netket/labels/help%20wanted).
These are issues that we believe are particularly well suited for outside
contributions. If you
decide to start on an issue, leave a comment so that other people know that
you're working on it. If you want to help out, but not alone, use the issue
comment thread to coordinate.

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

#### C++ coding style

NetKet C++ generally follows the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). The most evident exception is that in NetKet header files end in `.hpp` instead of the suggested `.h` as per
Google C++ Style.

We strongly recommend that you use `clang-tidy` to check your C/C++ changes.
To install clang-tidy on ubuntu, do:

```bash
apt-get install -y clang-tidy
```

on MacOS you can use `brew`

```bash
brew install clang-format
```

You can check a C/C++ file by doing:


```bash
clang-format --style=google <my_cc_file> > /tmp/my_cc_file.cc
diff <my_cc_file> /tmp/my_cc_file.cc
```

Most likely, your favorite code editor supports a plugin for `clang-tidy`.
Those are typically very handy, since they allow to format the code in real time,
at any point you save your code, and without recurring to the command line.


#### Running sanity check

You can run sanity checks compiling with the option `NETKET_Sanitizer=ON` in CMake:
```bash
mkdir build
cd build
cmake .. -DNETKET_Sanitizer=ON
make
make test
```

This will compile NetKet with Clang sanitizer, and unit tests will report any issue found.


#### Including new unit tests

New contributions **must** include associated unit tests.
Unit tests in NetKet are based on [Catch 2](https://github.com/catchorg/Catch2) and are in the directory `Test`.

In the most typical case, your contribution will be an extension of one of the many existing prototype classes (for example, deriving from `AbstractMachine`, `AbstractGraph` classes etc). In this case, you typically only need to add
a corresponding input file for testing.

For example, you can take a look at `Test/Machine/machine_input_tests.hpp`. If you add a new `Machine`, you must include in this file a json object which allows NetKet to construct your machine in a few test cases. All the units tests conceived for the `Machine` class will be then automatically executed on your new class.

#### Running unit tests
Unit tests are automatically compiled when you
install NetKet through CMake. To execute unit tests, just do

```bash
make test
```
in the CMake build directory.  

Notice that if you add new test files (i.e. you do not simply extend the existing json input objects), before compiling you will need to tell CMake about your new files, editing `Test/CMakeLists.txt`. 
