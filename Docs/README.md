# NetKet documentation

## Making All Docs
To generate all documentation for the modules listed in ``build_docs.py`` just run

```
python build_docs.py
```

This script will output a markdown file for each submodule and place it in a
directory that matches the NetKet project hierarchy. For example,
documentation for `netket.graph.hypercube` will be placed in `graph/hypercube.md`,
that resides in the root of this directory.

If a new submodule is added to main netket module, it should be manually added
in `build_docs.py`, appending its name to the list
`default_submodules=[graph,...]`.


## Making Docs for a Single Module
Documentation for individual modules can be built as follows (__Note: the
modules built this way do not have to be listed in `default_submodules`__):

```
python build_docs.py <netket_module>
```

Where `<netket_module>` is one of the main modules, such as `graph` or `hilbert`.

## Making Docs for a Single Class
It is also possible to generate documentation for a single class via

```
python make_class_docs.py <netket_class> 
```

For example, `make_class_docs.py netket.graph.Hypercube` will generate
documentation for the class `Hypercube` and write its contents to stdout.
