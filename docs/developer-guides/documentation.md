# Documentation

## Update documentation

To rebuild the documentation, run:

```
uv run make -C docs html
```

This can take a long time because it executes many of the notebooks in the documentation source;
if you'd prefer to build the docs without executing the notebooks, you can run:

```
uv run make -C docs html SPHINXOPTS="-D nb_execution_mode=off"
```

You can then see the generated documentation in `docs/_build/html/index.html`.

To clean the documentation build:

```
uv run make -C docs clean
```

(update-notebooks)=

## Update notebooks

We use [jupytext](https://jupytext.readthedocs.io/) to maintain two synced copies of the notebooks
in `docs/tutorials` and other documentation directories: one in `ipynb` format, and one in `md` format. The advantage of the former
is that it can be opened and executed directly in Colab; the advantage of the latter is that
it makes it much easier to track diffs within version control.

### Editing `ipynb`

For making large changes that substantially modify code and outputs, it is easiest to
edit the notebooks in Jupyter or in Colab. To edit notebooks in the Colab interface,
open <http://colab.research.google.com> and `Upload` from your local repo.
Update it as needed, `Run all cells` then `Download ipynb`.
You may want to test that it executes properly, using `sphinx-build` as explained above.

### Editing `md`

For making smaller changes to the text content of the notebooks, it is easiest to edit the
`.md` versions using a text editor.

### Syncing notebooks

After editing either the ipynb or md versions of the notebooks, you can sync the two versions
using [jupytext](https://jupytext.readthedocs.io/) by running `jupytext --sync` on the updated
notebooks; for example:

```
uv run jupytext --sync docs/tutorials/gs-ising.ipynb
```

### Regenerating notebook outputs

To regenerate a notebook's outputs after editing the Python version and ensure it runs correctly:

```
uv run jupytext --sync --execute docs/tutorials/gs-ising.ipynb
```

The jupytext version should match that specified in
[.pre-commit-config.yaml](https://github.com/netket/netket/blob/main/.pre-commit-config.yaml).

To check that the markdown and ipynb files are properly synced, you may use the
[pre-commit](https://pre-commit.com/) framework to perform the same check used
by the github CI:

```
uv run pre-commit run jupytext --all-files
```

### Creating new notebooks

If you are adding a new notebook to the documentation and would like to use the `jupytext --sync`
command discussed here, you can set up your notebook for jupytext by using the following command:

```
uv run jupytext --set-formats ipynb,md:myst path/to/the/notebook.ipynb
```

This works by adding a `"jupytext"` metadata field to the notebook file which specifies the
desired formats, and which the `jupytext --sync` command recognizes when invoked.