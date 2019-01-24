# NetKet documentation

## Making All Docs
To generate all documentation for the modules listed in ``build_docs.py`` just run

```bash
python build_docs.py
```

if a new submodule is added to main netket module, it should be manually added
in `build_docs.py`, appending its name to the list
`default_submodules=[graph,...]`.


## Making Docs for a Single Module
Documentation for individual modules can be built as follows (__Note: the
modules built this way do not have to be listed in `default_submodules`__):

```
python build_docs.py <netket_module>
```

Where `<netket_module>` is one of the main modules, such as `graph` or `hilbert`.
