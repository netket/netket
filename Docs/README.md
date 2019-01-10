# NetKet documentation
To generate documentation just do

```
python make_all_docs.py
```

if a new submodule is added to main netket module, it must be manually added in `make_all_docs.py`,
appending its name to the list `submodules=[graph,...]`
