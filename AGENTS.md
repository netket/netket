# AGENTS instructions for Codex

This repository follows the contribution guidelines described in `docs/contributing.html` (source: `docs/docs/contributing.md`). Key points:

- **Code style**: follow PEP 8 and format Python files with **Black**. Run `black` on modified files or `pre-commit run --files <paths>` before committing.
- **Linting**: run `ruff check --fix` to fix simple issues.
- **Docstrings & type hints**: document new functions or methods using Google style docstrings and add type hints. Use types from `typing` and `netket.utils.types` where appropriate.
- **Public API**: if you add user-facing objects, document them in `docs/api.rst` and create documentation pages as needed.
- **Testing**: run `pytest -n auto test/` to execute the test suite. If your changes touch MPI functionality, also run `mpirun -np 2 pytest -n0 test/`.
- **Commits & PRs**: keep commits self-contained and descriptive. PRs should ideally contain a single commit.
- **Pre-commit**: install the hooks with `pre-commit install` and run `pre-commit run --all-files` before submitting a PR.

For advice on writing tests (including MPI-aware tests), see `docs/docs/writing-tests.md`.
