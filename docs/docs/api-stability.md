# API Stability

NetKet follows [semantic versioning](https://semver.org) to indicate API stability.

Under semantic versioning, the version of a software is indicated by the tuple `X.Y.Z`.
 - X is the major version. Breaking changes to the stable API are only made when increasing
 the major version number.
 - Y is the minor version. This version number is increased when new features that do not
 break the stable API are introduced or when some features or APIs are deprecated (meaning
 that they will keep working but raise a warning asking the user to change the code)
 - Z is the patch version, and is increased when bug fixes are issued.

Please note that the API stability is guaranteed only for a subset of the whole API, usually
called the stable or public API.

This means that we *guarantee* that code written for NetKet 3.0 using the stable APIs will 
keep working for any other 3.X version, and might require changes only when Netket 4 will 
be released.

## Public API Definition

The parts of the API covered by the API stability guarantee and semantic versioning are all
methods, functions and classes listed in the [API reference](api) *unless stated otherwise 
in the docstring of said function or method*.

In general we try to ensure that all functions, methods and classes that can be accessed by 
dot-syntax and *with a name not starting with an underscore* are part of the public-API, 
such as in the example below.

```python
import netket 

# public api
netket.submodule.subsubmodule.public_fun

# private api
netket.submodule.subsubmodule._private_fun
```

```{warning}
If you only use functions and properties without a leading underscore, and follow
the tutorials and examples, your code will be fine.

If your code uses properties or functions with a leading underscore, you're 
probably using private, unstable APIs. That's ok, but be careful in not updating
the minor NetKet version!
```

Exceptions to the above rule are:

 - methods, functions or classes whose name starts with an underscore;
 - methods, functions, classes or modules with `experimental` in their name or
 inside an experimental module;
 - methods, functions, classes or modules inside of `netket.utils`, unless 
 documented in the api reference. This module is mainly for internal use by netket;
 - The precise type of an object: while we guarantee that the code `class_name(args)`
 will keep working in future versions of NetKet we might substitute `class_name` by
 a function returning the same class as before;
 - The class hierarchy: we might insert new abstract classes in the class hierarchy 
 of existing classes;
 - The precise type returned by most functions, methods and constructors;
 - The print/repr representation of NetKet objects;
 - API deprecated in NetKet 3.0: NetKet 3.0 contains some already-deprecated API.
 Those will be removed in a few months time with the release of NetKet 3.1. Those are
 remnants of experimental apis introduced during the beta development of NetKet 3 and
 we are leaving them in as a courtesy to those that helped us by testing the beta versions.
 However, to streamline the development, they will be removed within this major release cycle.
 Please keep an eye for deprecation warnings!


Exceptions also include those modules that will be refactored in the next minor version:

 - The API needed to define a new sampler might incur minor changes in NetKet 3.1, however
 we will try to avoid breaking user code if possible;
 - The API needed to define new operators will incur *major* breaking changes in NetKet 
 3.1/3.2 and is therefore not reccomended to define new operators. The user-facing API
 of operators is instead stable;

## Netket.nn module

The `netket.nn` module mirrors some functions found in `jax.nn` and `flax.linen`, while
addressing some minor/major bugs, usually in the way they interact with complex numbers.
We are hard at work upstreaming those changes to the original libraries, so that other 
users can profit of our bugfixes.
For this reason, during the NetKet 3.X life-cycle we will gradually deprecate functionality
found within `netket.nn` and suggest you upgrade your jax/flax version and use the functionality
found in those packages.

Our plan is that in NetKet 4.X `netket.nn` will only contain NetKet-specific Neural-Network
structures, and we might even split it off into it's own package.

For this reason, since both `jax` and `flax` have their own versioning scheme, we reccomend
you take care in versioning your environment, that is, you check their version too and 
try to keep the same minor version of those libraries in your project.


## Storing the environment

We strongly reccomend that you store somewhere the exact version number, or at least the
`major.minor` version number you used in a project so that you can reproduce it later on.
We also strongly advise to keep the version numbers of *every python package installed*, in
particoular of `jax`, `jaxlib`, `flax` and `optax` because they might not be as precise as
us when updating their version numbers.

An easy way to achieve this is to use a package manager such as [poetry](https://python-poetry.org) 
or register version numbers with the command.

```bash
pip freeze >> requirements.txt
```

