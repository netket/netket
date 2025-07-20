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

To edit notebooks, you can use Jupyter or Colab. To edit notebooks in the Colab interface,
open <http://colab.research.google.com> and `Upload` from your local repo.
Update it as needed, `Run all cells` then `Download ipynb`.
You may want to test that it executes properly, using `sphinx-build` as explained above.