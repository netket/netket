# NetKet Documentation

To build the documentation run

```
uv run make clean
uv run make html
```

## Notebooks

Notebooks are managed with `jupytext`, and commited to the git repository in both their `.py` form (which does not include output of cells) and in their `.ipynb` form.

Look at the developer documentation for full details but, if you edit one of the two you should run

```
uv run jupytext --sync path/to/edited/file.{ipynb/py}
```

And to regenerate the output cells of the notebook run
```
uv run jupytext --sync --execute path/to/edited/file.{ipynb/py}
```